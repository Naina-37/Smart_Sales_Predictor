
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from bson import ObjectId
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import os
import io
import random
import smtplib
from email.message import EmailMessage
import pandas as pd
from io import StringIO

from src.model import load_model
from src.preprocess import load_and_clean_data
from src.simulator import predict_sales
import random
import smtplib

from email.mime.text import MIMEText
from fastapi import Body


# ================= APP SETUP =================

load_dotenv()

app = FastAPI(title="Smart Sales Intelligence Platform - Enterprise API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================= SECURITY CONFIG =================

SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret-key")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# ================= DATABASE =================

MONGO_URL = os.getenv("MONGO_URL")
print("MONGO_URL loaded:", MONGO_URL)

if not MONGO_URL:
    raise ValueError("MONGO_URL not found. Check .env file.")

client = MongoClient(MONGO_URL)
db = client["smart_sales_predictor"]

users_collection = db["users"]
folders_collection = db["folders"]
products_collection = db["products"]
customers_collection = db["customers"]
inventory_collection = db["inventory"]
history_collection = db["prediction_history"]
csv_collection = db["csv_predictions"]
activity_collection = db["activity_logs"]
notifications_collection = db["notifications"]
otp_collection = db["otp_codes"]


# ================= ML MODEL =================

model = load_model()
data = load_and_clean_data("data/sales.csv")
X = data.drop(["units_sold", "date", "demand_forecast"], axis=1)
X = X.select_dtypes(include=["int64", "float64"])


# ================= MODELS =================

class UserInput(BaseModel):
    username: str
    password: str
    email: str = ""
    phone: str = ""
    role: str = "user"

class LoginInput(BaseModel):
    username: str
    password: str

class GoogleLoginInput(BaseModel):
    credential: str

class RoleUpdate(BaseModel):
    username: str
    role: str

class FolderInput(BaseModel):
    folder_name: str
    category: str = "General"

class ProductInput(BaseModel):
    product_name: str
    category: str = "General"
    region: str = "India"
    folder_name: str = "General"
    price: float
    discount: float = 0
    promotion: int = 0
    stock_quantity: int = 0
    supplier: str = "Not Provided"

class CustomerInput(BaseModel):
    customer_name: str
    phone: str = ""
    email: str = ""
    product_name: str
    quantity: int
    purchase_amount: float
    region: str = "India"

class InventoryInput(BaseModel):
    product_name: str
    category: str = "General"
    stock_quantity: int
    reorder_level: int = 10
    supplier: str = "Not Provided"
    folder_name: str = "General"

class PredictionInput(BaseModel):
    price: float
    discount: float
    promotion: int
    product_name: str = "General Product"
    category: str = "General"
    region: str = "India"
    folder_name: str = "General"

class OTPRequest(BaseModel):
    identifier: str
    purpose: str = "register"  # register, reset_password, login

class OTPVerify(BaseModel):
    identifier: str
    otp: str
    purpose: str = "register"

class ResetPasswordInput(BaseModel):
    identifier: str
    otp: str
    new_password: str


# ================= SECURITY HELPERS =================

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = users_collection.find_one({"username": username})
    if user is None:
        raise credentials_exception
    return user

def require_roles(allowed_roles: list):
    def role_checker(current_user: dict = Depends(get_current_user)):
        role = current_user.get("role", "user")
        if role not in allowed_roles:
            raise HTTPException(status_code=403, detail="You do not have permission to access this feature.")
        return current_user
    return role_checker


# ================= GENERAL HELPERS =================

def obj_to_str(record):
    record["_id"] = str(record["_id"])
    if "created_at" in record and isinstance(record["created_at"], datetime):
        record["created_at"] = record["created_at"].strftime("%Y-%m-%d %H:%M:%S")
    if "updated_at" in record and isinstance(record["updated_at"], datetime):
        record["updated_at"] = record["updated_at"].strftime("%Y-%m-%d %H:%M:%S")
    record.pop("password", None)
    return record

def log_activity(username, action, details=""):
    activity_collection.insert_one({
        "username": username,
        "action": action,
        "details": details,
        "created_at": datetime.now()
    })

def add_notification(username, title, message, notification_type="info"):
    notifications_collection.insert_one({
        "username": username,
        "title": title,
        "message": message,
        "type": notification_type,
        "is_read": False,
        "created_at": datetime.now()
    })

def notify_admins(title, message, notification_type="info"):
    admins = list(users_collection.find({"role": "admin"}))

    if not admins:
        add_notification("admin", title, message, notification_type)
        return

    for admin in admins:
        add_notification(
            admin.get("username", "admin"),
            title,
            message,
            notification_type
        )

def check_low_stock(product_name, stock_quantity, reorder_level):
    if stock_quantity <= reorder_level:
        notify_admins(
            "Low Stock Alert",
            f"{product_name} stock is low. Current stock: {stock_quantity}, reorder level: {reorder_level}.",
            "warning"
        )
        return True

    return False

def send_email(to_email: str, subject: str, body: str):
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_email = os.getenv("SMTP_EMAIL")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not smtp_host or not smtp_email or not smtp_password:
        print(f"DEV EMAIL to {to_email}: {subject}\n{body}")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_email
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_email, smtp_password)
        server.send_message(msg)
    return True

def generate_otp():
    return str(random.randint(100000, 999999))

def generate_prediction(price, discount, promotion):
    sample = X.iloc[0].copy()
    sample["price"] = price
    sample["discount"] = discount
    sample["holiday_promotion"] = promotion
    return int(predict_sales(model, sample, X.columns))

def calculate_factor_breakdown(price, discount, promotion):
    base_effect = max(price * 0.40, 1)
    discount_effect = max(price * (discount / 100) * 0.70, 0)
    promotion_effect = price * 0.25 if promotion == 1 else 0
    total = base_effect + discount_effect + promotion_effect
    return {
        "base_effect": round(base_effect, 2),
        "discount_effect": round(discount_effect, 2),
        "promotion_effect": round(promotion_effect, 2),
        "base_percent": round((base_effect / total) * 100, 2),
        "discount_percent": round((discount_effect / total) * 100, 2),
        "promotion_percent": round((promotion_effect / total) * 100, 2)
    }

def generate_prediction_insight(product_name, category, region, price, discount, promotion, prediction):
    insight = f"For {product_name} in {region}, predicted sales are ₹{prediction}. "
    insight += "Promotion is active, which may increase attention and demand. " if promotion == 1 else "Promotion is inactive, so sales depend more on price and discount. "
    if discount >= 30:
        insight += "Discount is high; demand may rise but profit margin can reduce. "
    elif discount >= 10:
        insight += "Discount is balanced and can support healthy demand. "
    else:
        insight += "Discount is low; consider promotion or better pricing. "
    if price > 1000:
        insight += "Price is high, so customers may be more price-sensitive. "
    elif price < 100:
        insight += "Low price can attract budget-conscious customers. "
    else:
        insight += "Moderate price supports stable demand. "
    insight += f"{category} should be monitored for stock, price, and promotion planning."
    return insight


# ================= BASIC ROUTES =================

@app.get("/")
def home():
    return {"message": "Smart Sales Intelligence Platform Enterprise API running"}

@app.get("/me")
def me(current_user: dict = Depends(get_current_user)):
    return {
        "username": current_user["username"],
        "email": current_user.get("email", ""),
        "phone": current_user.get("phone", ""),
        "role": current_user.get("role", "user"),
        "status": current_user.get("status", "active"),
        "auth_provider": current_user.get("auth_provider", "manual"),
        "picture": current_user.get("google_picture", "")
    }


# ================= OTP + EMAIL =================

@app.post("/send-otp")
def send_otp(data: OTPRequest):
    otp = generate_otp()
    expires_at = datetime.now() + timedelta(minutes=10)

    otp_collection.insert_one({
        "identifier": data.identifier,
        "otp": otp,
        "purpose": data.purpose,
        "is_used": False,
        "expires_at": expires_at,
        "created_at": datetime.now()
    })

    if "@" in data.identifier:
        send_email(
            data.identifier,
            "Smart Sales Predictor OTP",
            f"Your OTP for {data.purpose} is {otp}. It is valid for 10 minutes."
        )
        return {"message": "OTP sent to email", "dev_otp": otp}

    # SMS provider can be integrated later using Twilio/Fast2SMS.
    print(f"DEV SMS OTP for {data.identifier}: {otp}")
    return {"message": "OTP generated for phone. SMS gateway pending.", "dev_otp": otp}

@app.post("/verify-otp")
def verify_otp(data: OTPVerify):
    record = otp_collection.find_one({
        "identifier": data.identifier,
        "otp": data.otp,
        "purpose": data.purpose,
        "is_used": False
    })

    if not record:
        return {"error": "Invalid OTP"}

    if record["expires_at"] < datetime.now():
        return {"error": "OTP expired"}

    otp_collection.update_one({"_id": record["_id"]}, {"$set": {"is_used": True}})
    return {"message": "OTP verified successfully"}

@app.post("/reset-password")
def reset_password(data: ResetPasswordInput):
    otp_record = otp_collection.find_one({
        "identifier": data.identifier,
        "otp": data.otp,
        "purpose": "reset_password",
        "is_used": False
    })

    if not otp_record:
        return {"error": "Invalid OTP"}

    if otp_record["expires_at"] < datetime.now():
        return {"error": "OTP expired"}

    query = {"$or": [{"username": data.identifier}, {"email": data.identifier}, {"phone": data.identifier}]}
    user = users_collection.find_one(query)
    if not user:
        return {"error": "User not found"}

    users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {"password": hash_password(data.new_password), "auth_provider": "manual"}}
    )
    otp_collection.update_one({"_id": otp_record["_id"]}, {"$set": {"is_used": True}})
    add_notification(user["username"], "Password Reset", "Your password was reset successfully.", "success")
    return {"message": "Password reset successful"}


# ================= AUTH ROUTES =================

@app.post("/register")
def register(user: UserInput):
    existing_user = users_collection.find_one({
        "$or": [
            {"username": user.username},
            {"email": user.email} if user.email else {"username": "__never__"},
            {"phone": user.phone} if user.phone else {"username": "__never__"}
        ]
    })

    if existing_user:
        return {"error": "User already exists"}

    allowed_roles = ["admin", "sales_manager", "user"]
    total_users = users_collection.count_documents({})

    if total_users == 0:
        role = "admin"
    else:
        role = user.role if user.role in allowed_roles else "user"
        if role == "admin":
            role = "user"

    users_collection.insert_one({
        "username": user.username,
        "email": user.email,
        "phone": user.phone,
        "password": hash_password(user.password),
        "auth_provider": "manual",
        "role": role,
        "status": "active",
        "email_verified": False,
        "phone_verified": False,
        "created_at": datetime.now()
    })

    if user.email:
        send_email(
            user.email,
            "Welcome to Smart Sales Predictor",
            f"Hi {user.username}, your account has been created successfully with role: {role}."
        )

    log_activity(user.username, "Registered", f"User registered as {role}")
    add_notification(user.username, "Welcome", "Your account has been created successfully.", "success")

    return {"message": "Registered successfully", "username": user.username, "role": role}

@app.post("/login")
def login(user: LoginInput):
    existing_user = users_collection.find_one({
        "$or": [
            {"username": user.username},
            {"email": user.username},
            {"phone": user.username}
        ]
    })

    if not existing_user:
        return {"error": "Invalid username or password"}

    if existing_user.get("auth_provider") == "google" and not existing_user.get("password"):
        return {"error": "This account uses Google login. Please continue with Google."}

    if not verify_password(user.password, existing_user["password"]):
        return {"error": "Invalid username or password"}

    access_token = create_access_token({"sub": existing_user["username"], "role": existing_user.get("role", "user")})
    log_activity(existing_user["username"], "Login", "Manual login successful")

    return {
        "message": "Login successful",
        "access_token": access_token,
        "token_type": "bearer",
        "username": existing_user["username"],
        "role": existing_user.get("role", "user")
    }

@app.post("/auth/google")
def google_auth(data: GoogleLoginInput):
    try:
        if not GOOGLE_CLIENT_ID:
            return {"error": "Google Client ID is not configured in backend .env"}

        idinfo = id_token.verify_oauth2_token(data.credential, google_requests.Request(), GOOGLE_CLIENT_ID)
        email = idinfo.get("email")
        name = idinfo.get("name", "")
        picture = idinfo.get("picture", "")

        if not email:
            return {"error": "Google account email not found"}

        existing_user = users_collection.find_one({"email": email})
        total_users = users_collection.count_documents({})

        if existing_user:
            role = existing_user.get("role", "user")
            username = existing_user["username"]
        else:
            role = "admin" if total_users == 0 else "user"
            username_base = name.replace(" ", "_").lower() if name else email.split("@")[0]
            username = username_base
            counter = 1
            while users_collection.find_one({"username": username}):
                username = f"{username_base}_{counter}"
                counter += 1

            users_collection.insert_one({
                "username": username,
                "email": email,
                "phone": "",
                "password": None,
                "auth_provider": "google",
                "google_picture": picture,
                "role": role,
                "status": "active",
                "email_verified": True,
                "phone_verified": False,
                "created_at": datetime.now()
            })
            send_email(email, "Welcome to Smart Sales Predictor", f"Hi {username}, your Google account login is ready.")

        access_token = create_access_token({"sub": username, "role": role})
        log_activity(username, "Login", "Google login successful")

        return {
            "message": "Google login successful",
            "access_token": access_token,
            "token_type": "bearer",
            "username": username,
            "email": email,
            "role": role,
            "picture": picture
        }

    except ValueError:
        return {"error": "Invalid Google token"}


# ================= USER MANAGEMENT =================

@app.get("/users")
def get_users(current_user: dict = Depends(require_roles(["admin"]))):
    users = list(users_collection.find().sort("created_at", -1))
    return {"users": [obj_to_str(user) for user in users]}

@app.post("/update-role")
def update_role(data: RoleUpdate, current_user: dict = Depends(require_roles(["admin"]))):
    allowed_roles = ["admin", "sales_manager", "user"]
    if data.role not in allowed_roles:
        return {"error": "Invalid role"}
    result = users_collection.update_one({"username": data.username}, {"$set": {"role": data.role}})
    if result.matched_count == 0:
        return {"error": "User not found"}
    log_activity(current_user["username"], "Role Updated", f"{data.username} changed to {data.role}")
    return {"message": "Role updated successfully"}

@app.delete("/delete-user/{username}")
def delete_user(username: str, current_user: dict = Depends(require_roles(["admin"]))):
    if username == current_user["username"]:
        return {"error": "Admin cannot delete own account"}
    result = users_collection.delete_one({"username": username})
    if result.deleted_count == 0:
        return {"error": "User not found"}
    log_activity(current_user["username"], "User Deleted", username)
    return {"message": "User deleted successfully"}


# ================= FOLDERS =================

@app.post("/folders")
def create_folder(folder: FolderInput, current_user: dict = Depends(require_roles(["admin", "sales_manager"]))):
    if folders_collection.find_one({"folder_name": folder.folder_name}):
        return {"error": "Folder already exists"}
    folders_collection.insert_one({
        "folder_name": folder.folder_name,
        "category": folder.category,
        "created_by": current_user["username"],
        "created_at": datetime.now()
    })
    log_activity(current_user["username"], "Folder Created", folder.folder_name)
    return {"message": "Folder created successfully"}

@app.get("/folders")
def get_folders(current_user: dict = Depends(get_current_user)):
    folders = list(folders_collection.find().sort("created_at", -1))
    return {"folders": [obj_to_str(folder) for folder in folders]}

@app.delete("/folders/{folder_name}")
def delete_folder(folder_name: str, current_user: dict = Depends(require_roles(["admin"]))):
    folders_collection.delete_one({"folder_name": folder_name})
    log_activity(current_user["username"], "Folder Deleted", folder_name)
    return {"message": "Folder deleted successfully"}


# ================= PRODUCTS =================

@app.post("/products")
def add_product(product: ProductInput, current_user: dict = Depends(require_roles(["admin", "sales_manager"]))):
    products_collection.insert_one({
        **product.dict(),
        "created_by": current_user["username"],
        "created_at": datetime.now()
    })
    log_activity(current_user["username"], "Product Added", product.product_name)
    return {"message": "Product added successfully"}

@app.get("/products")
def get_products(
    search: str = "",
    category: str = "",
    folder_name: str = "",
    current_user: dict = Depends(get_current_user)
):
    query = {}
    if search:
        query["product_name"] = {"$regex": search, "$options": "i"}
    if category:
        query["category"] = {"$regex": category, "$options": "i"}
    if folder_name:
        query["folder_name"] = {"$regex": folder_name, "$options": "i"}

    products = list(products_collection.find(query).sort("created_at", -1))
    return {"products": [obj_to_str(product) for product in products]}

@app.get("/products/{folder_name}")
def get_products_by_folder(folder_name: str, current_user: dict = Depends(get_current_user)):
    products = list(products_collection.find({"folder_name": folder_name}).sort("created_at", -1))
    return {"products": [obj_to_str(product) for product in products]}

@app.delete("/products/{product_id}")
def delete_product(product_id: str, current_user: dict = Depends(require_roles(["admin", "sales_manager"]))):
    products_collection.delete_one({"_id": ObjectId(product_id)})
    log_activity(current_user["username"], "Product Deleted", product_id)
    return {"message": "Product deleted successfully"}


# ================= CUSTOMERS =================

@app.post("/customers")
def add_customer(customer: CustomerInput, current_user: dict = Depends(require_roles(["admin", "sales_manager"]))):
    customers_collection.insert_one({
        **customer.dict(),
        "created_by": current_user["username"],
        "created_at": datetime.now()
    })
    log_activity(current_user["username"], "Customer Added", customer.customer_name)
    return {"message": "Customer record added successfully"}

@app.get("/customers")
def get_customers(search: str = "", current_user: dict = Depends(get_current_user)):
    query = {}
    if search:
        query["$or"] = [
            {"customer_name": {"$regex": search, "$options": "i"}},
            {"product_name": {"$regex": search, "$options": "i"}},
            {"region": {"$regex": search, "$options": "i"}}
        ]
    customers = list(customers_collection.find(query).sort("created_at", -1))
    return {"customers": [obj_to_str(customer) for customer in customers]}


# ================= INVENTORY =================

@app.post("/inventory")
def add_inventory(item: InventoryInput, current_user: dict = Depends(require_roles(["admin", "sales_manager"]))):
    inventory_collection.insert_one({
        **item.dict(),
        "created_by": current_user["username"],
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    })

    is_low_stock = check_low_stock(
        item.product_name,
        item.stock_quantity,
        item.reorder_level
    )

    log_activity(current_user["username"], "Inventory Added", item.product_name)

    if is_low_stock:
        log_activity(
            current_user["username"],
            "Low Stock Alert",
            f"{item.product_name} stock is below reorder level"
        )

    return {
        "message": "Inventory item added successfully",
        "low_stock": is_low_stock
    }

@app.get("/inventory")
def get_inventory(current_user: dict = Depends(get_current_user)):
    items = list(inventory_collection.find().sort("created_at", -1))
    data = []
    for item in items:
        item["stock_status"] = "Low Stock" if item.get("stock_quantity", 0) <= item.get("reorder_level", 10) else "Available"
        data.append(obj_to_str(item))
    return {"inventory": data}


# ================= PREDICTION =================

@app.post("/predict")
def predict(input_data: PredictionInput, current_user: dict = Depends(require_roles(["admin", "sales_manager"]))):
    prediction = generate_prediction(input_data.price, input_data.discount, input_data.promotion)
    breakdown = calculate_factor_breakdown(input_data.price, input_data.discount, input_data.promotion)
    insight = generate_prediction_insight(
        input_data.product_name, input_data.category, input_data.region,
        input_data.price, input_data.discount, input_data.promotion, prediction
    )

    history_collection.insert_one({
        **input_data.dict(),
        "predicted_sales": prediction,
        "insight": insight,
        "breakdown": breakdown,
        "source": "manual_prediction",
        "created_by": current_user["username"],
        "created_at": datetime.now()
    })

    log_activity(current_user["username"], "Prediction Created", input_data.product_name)

    return {
        "predicted_sales": prediction,
        "product_name": input_data.product_name,
        "category": input_data.category,
        "region": input_data.region,
        "folder_name": input_data.folder_name,
        "breakdown": breakdown,
        "insight": insight
    }


# ================= CSV BULK PREDICTION =================

@app.post("/csv-predict")
async def csv_predict(file: UploadFile = File(...), current_user: dict = Depends(require_roles(["admin", "sales_manager"]))):
    content = await file.read()
    csv_text = content.decode("utf-8")
    df = pd.read_csv(StringIO(csv_text))

    required_columns = ["price", "discount", "promotion"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return {"error": f"Missing required columns: {missing}"}

    results = []
    for index, row in df.iterrows():
        product_name = str(row.get("product_name", f"Product {index + 1}"))
        category = str(row.get("category", "General"))
        region = str(row.get("region", "India"))
        folder_name = str(row.get("folder_name", category))
        price = float(row["price"])
        discount = float(row["discount"])
        promotion = int(row["promotion"])

        prediction = generate_prediction(price, discount, promotion)
        breakdown = calculate_factor_breakdown(price, discount, promotion)
        insight = generate_prediction_insight(product_name, category, region, price, discount, promotion, prediction)

        result = {
            "row": index + 1,
            "product_name": product_name,
            "category": category,
            "region": region,
            "folder_name": folder_name,
            "price": price,
            "discount": discount,
            "promotion": promotion,
            "predicted_sales": prediction,
            "breakdown": breakdown,
            "insight": insight
        }
        results.append(result)

        history_collection.insert_one({
            **result,
            "source": "csv_upload",
            "created_by": current_user["username"],
            "created_at": datetime.now()
        })

    csv_collection.insert_one({
        "filename": file.filename,
        "total_rows": len(results),
        "results": results,
        "uploaded_by": current_user["username"],
        "created_at": datetime.now()
    })

    log_activity(current_user["username"], "CSV Uploaded", file.filename)
    return {"results": results}


# ================= HISTORY + ANALYTICS =================

@app.get("/history")
def get_history(
    product_name: str = "",
    category: str = "",
    region: str = "",
    folder_name: str = "",
    current_user: dict = Depends(get_current_user)
):
    query = {}
    if product_name:
        query["product_name"] = {"$regex": product_name, "$options": "i"}
    if category:
        query["category"] = {"$regex": category, "$options": "i"}
    if region:
        query["region"] = {"$regex": region, "$options": "i"}
    if folder_name:
        query["folder_name"] = {"$regex": folder_name, "$options": "i"}

    records = list(history_collection.find(query).sort("created_at", -1))
    return {"history": [obj_to_str(record) for record in records]}

@app.get("/analytics")
def analytics(current_user: dict = Depends(get_current_user)):
    records = list(history_collection.find())

    if not records:
        return {
            "total_predictions": 0,
            "average_sales": 0,
            "max_sales": 0,
            "min_sales": 0,
            "promotion_active_count": 0,
            "promotion_inactive_count": 0,
            "category_summary": [],
            "product_summary": [],
            "region_summary": [],
            "folder_summary": [],
            "chart_data": [],
            "ai_insights": ["No prediction data is available yet. Run predictions to generate analytics."]
        }

    sales_values = [r.get("predicted_sales", 0) for r in records]
    promotion_active = [r for r in records if r.get("promotion") == 1]
    promotion_inactive = [r for r in records if r.get("promotion") == 0]

    def grouped_summary(field_name, output_name):
        grouped = {}
        for record in records:
            key = record.get(field_name, "Unknown")
            grouped.setdefault(key, {output_name: key, "count": 0, "total_sales": 0})
            grouped[key]["count"] += 1
            grouped[key]["total_sales"] += record.get("predicted_sales", 0)

        return sorted([
            {
                output_name: item[output_name],
                "count": item["count"],
                "average_sales": round(item["total_sales"] / item["count"], 2)
            }
            for item in grouped.values()
        ], key=lambda x: x["average_sales"], reverse=True)

    category_summary = grouped_summary("category", "category")
    product_summary = grouped_summary("product_name", "product_name")
    region_summary = grouped_summary("region", "region")
    folder_summary = grouped_summary("folder_name", "folder_name")

    chart_data = [
        {
            "product_name": r.get("product_name", "Product"),
            "category": r.get("category", "General"),
            "region": r.get("region", "India"),
            "folder_name": r.get("folder_name", "General"),
            "price": r.get("price", 0),
            "discount": r.get("discount", 0),
            "promotion": r.get("promotion", 0),
            "predicted_sales": r.get("predicted_sales", 0)
        }
        for r in records[-20:]
    ]

    avg_sales = round(sum(sales_values) / len(sales_values), 2)
    max_sales = max(sales_values)
    min_sales = min(sales_values)

    active_avg = round(sum(r.get("predicted_sales", 0) for r in promotion_active) / len(promotion_active), 2) if promotion_active else 0
    inactive_avg = round(sum(r.get("predicted_sales", 0) for r in promotion_inactive) / len(promotion_inactive), 2) if promotion_inactive else 0

    ai_insights = []
    if product_summary:
        ai_insights.append(f"{product_summary[0]['product_name']} is the best-performing product with average predicted sales of ₹{product_summary[0]['average_sales']}.")
    if category_summary:
        ai_insights.append(f"{category_summary[0]['category']} is the strongest category with average predicted sales of ₹{category_summary[0]['average_sales']}.")
    if region_summary:
        ai_insights.append(f"{region_summary[0]['region']} is the highest-performing region based on current predictions.")
    if active_avg > inactive_avg:
        ai_insights.append(f"Promoted products are performing better. Average promoted sales are ₹{active_avg}, compared to ₹{inactive_avg} without promotion.")
    elif inactive_avg > active_avg:
        ai_insights.append(f"Non-promoted products are performing better. Average non-promoted sales are ₹{inactive_avg}, compared to ₹{active_avg} with promotion.")
    else:
        ai_insights.append("Promotion impact is neutral based on available data.")
    ai_insights.append(f"Overall average predicted sales are ₹{avg_sales}. Highest prediction is ₹{max_sales}, lowest prediction is ₹{min_sales}.")

    return {
        "total_predictions": len(records),
        "average_sales": avg_sales,
        "max_sales": max_sales,
        "min_sales": min_sales,
        "promotion_active_count": len(promotion_active),
        "promotion_inactive_count": len(promotion_inactive),
        "category_summary": category_summary,
        "product_summary": product_summary,
        "region_summary": region_summary,
        "folder_summary": folder_summary,
        "chart_data": chart_data,
        "ai_insights": ai_insights
    }


# ================= ADMIN SUMMARY + LOGS =================

@app.get("/admin-summary")
def admin_summary(current_user: dict = Depends(require_roles(["admin"]))):
    return {
        "total_users": users_collection.count_documents({}),
        "total_folders": folders_collection.count_documents({}),
        "total_products": products_collection.count_documents({}),
        "total_customers": customers_collection.count_documents({}),
        "total_inventory_items": inventory_collection.count_documents({}),
        "total_predictions": history_collection.count_documents({}),
        "total_csv_uploads": csv_collection.count_documents({})
    }

@app.get("/activity-logs")
def activity_logs(current_user: dict = Depends(require_roles(["admin"]))):
    logs = list(activity_collection.find().sort("created_at", -1).limit(100))
    return {"activity_logs": [obj_to_str(log) for log in logs]}

@app.get("/notifications")
def notifications(current_user: dict = Depends(get_current_user)):
    notes = list(notifications_collection.find({"username": current_user["username"]}).sort("created_at", -1).limit(50))
    return {"notifications": [obj_to_str(note) for note in notes]}


# ================= ENTERPRISE DASHBOARD + INSIGHTS =================

@app.get("/dashboard-stats")
def dashboard_stats(current_user: dict = Depends(get_current_user)):
    low_stock_count = 0
    inventory_items = list(inventory_collection.find())

    for item in inventory_items:
        if item.get("stock_quantity", 0) <= item.get("reorder_level", 10):
            low_stock_count += 1

    recent_prediction = history_collection.find_one(sort=[("created_at", -1)])

    return {
        "total_users": users_collection.count_documents({}),
        "total_folders": folders_collection.count_documents({}),
        "total_products": products_collection.count_documents({}),
        "total_customers": customers_collection.count_documents({}),
        "total_inventory_items": inventory_collection.count_documents({}),
        "low_stock_items": low_stock_count,
        "total_predictions": history_collection.count_documents({}),
        "total_csv_uploads": csv_collection.count_documents({}),
        "latest_prediction": obj_to_str(recent_prediction) if recent_prediction else None
    }

@app.get("/top-products")
def top_products(current_user: dict = Depends(get_current_user)):
    records = list(history_collection.find())

    summary = {}

    for record in records:
        product = record.get("product_name", "Unknown Product")
        predicted_sales = record.get("predicted_sales", 0)

        if product not in summary:
            summary[product] = {
                "product_name": product,
                "total_predicted_sales": 0,
                "prediction_count": 0
            }

        summary[product]["total_predicted_sales"] += predicted_sales
        summary[product]["prediction_count"] += 1

    result = []
    for item in summary.values():
        item["average_predicted_sales"] = round(
            item["total_predicted_sales"] / item["prediction_count"],
            2
        )
        result.append(item)

    result = sorted(
        result,
        key=lambda x: x["total_predicted_sales"],
        reverse=True
    )

    return {"top_products": result[:10]}

@app.get("/customer-insights")
def customer_insights(current_user: dict = Depends(get_current_user)):
    customers = list(customers_collection.find())

    total_customers = len(customers)
    total_purchase_amount = sum(customer.get("purchase_amount", 0) for customer in customers)
    average_purchase_amount = round(total_purchase_amount / total_customers, 2) if total_customers else 0

    region_summary = {}
    product_summary = {}

    for customer in customers:
        region = customer.get("region", "Unknown")
        product = customer.get("product_name", "Unknown Product")
        amount = customer.get("purchase_amount", 0)

        if region not in region_summary:
            region_summary[region] = {"region": region, "customers": 0, "total_purchase": 0}

        region_summary[region]["customers"] += 1
        region_summary[region]["total_purchase"] += amount

        if product not in product_summary:
            product_summary[product] = {"product_name": product, "customers": 0, "total_purchase": 0}

        product_summary[product]["customers"] += 1
        product_summary[product]["total_purchase"] += amount

    top_customers = sorted(
        customers,
        key=lambda x: x.get("purchase_amount", 0),
        reverse=True
    )[:5]

    return {
        "total_customers": total_customers,
        "total_purchase_amount": round(total_purchase_amount, 2),
        "average_purchase_amount": average_purchase_amount,
        "top_customers": [obj_to_str(customer) for customer in top_customers],
        "region_summary": list(region_summary.values()),
        "product_summary": list(product_summary.values())
    }

@app.get("/recent-activities")
def recent_activities(current_user: dict = Depends(get_current_user)):
    activities = list(
        activity_collection.find()
        .sort("created_at", -1)
        .limit(15)
    )

    return {"activities": [obj_to_str(activity) for activity in activities]}

@app.get("/report-summary")
def report_summary(current_user: dict = Depends(get_current_user)):
    analytics_data = analytics(current_user)
    dashboard_data = dashboard_stats(current_user)
    customer_data = customer_insights(current_user)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generated_by": current_user.get("username", "user"),
        "dashboard": dashboard_data,
        "analytics": analytics_data,
        "customers": customer_data
    }

@app.get("/inventory-alerts")
def inventory_alerts(current_user: dict = Depends(get_current_user)):
    items = list(inventory_collection.find())
    low_stock_items = []

    for item in items:
        stock_quantity = item.get("stock_quantity", 0)
        reorder_level = item.get("reorder_level", 10)

        if stock_quantity <= reorder_level:
            item["stock_status"] = "Low Stock"
            item["recommendation"] = f"Reorder {item.get('product_name', 'this product')} soon."
            low_stock_items.append(obj_to_str(item))

    return {
        "low_stock_count": len(low_stock_items),
        "low_stock_items": low_stock_items
    }

@app.post("/notifications/{notification_id}/read")
def mark_notification_read(notification_id: str, current_user: dict = Depends(get_current_user)):
    try:
        result = notifications_collection.update_one(
            {"_id": ObjectId(notification_id)},
            {"$set": {"is_read": True}}
        )

        if result.matched_count == 0:
            return {"error": "Notification not found"}

        return {"message": "Notification marked as read"}

    except Exception as e:
        return {"error": str(e)}

@app.get("/notifications/unread-count")
def unread_notification_count(current_user: dict = Depends(get_current_user)):
    count = notifications_collection.count_documents({
        "username": current_user["username"],
        "is_read": False
    })

    return {"unread_count": count}

# ================= EXPORTS =================

@app.get("/export/history-csv")
def export_history_csv(current_user: dict = Depends(get_current_user)):
    records = list(history_collection.find())
    rows = []
    for r in records:
        r.pop("_id", None)
        r.pop("breakdown", None)
        rows.append(r)

    df = pd.DataFrame(rows)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=prediction_history.csv"}
    )

@app.get("/export/history-excel")
def export_history_excel(current_user: dict = Depends(get_current_user)):
    records = list(history_collection.find())
    rows = []
    for r in records:
        r.pop("_id", None)
        r.pop("breakdown", None)
        rows.append(r)

    df = pd.DataFrame(rows)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Prediction History")
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=prediction_history.xlsx"}
    )
