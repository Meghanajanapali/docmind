import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Load variables from a local .env file (never commit this file)
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Create a .env file based on .env.example "
        "and set DATABASE_URL there."
    )

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(bind=engine)