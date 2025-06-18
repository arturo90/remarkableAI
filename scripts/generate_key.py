import secrets
import base64

def generate_secret_key():
    """Generate a secure random key suitable for use as SECRET_KEY."""
    # Generate 32 random bytes and encode them in base64
    random_bytes = secrets.token_bytes(32)
    secret_key = base64.b64encode(random_bytes).decode('utf-8')
    return secret_key

if __name__ == "__main__":
    key = generate_secret_key()
    print("\nGenerated SECRET_KEY:")
    print(f"SECRET_KEY={key}")
    print("\nCopy this key to your .env file") 