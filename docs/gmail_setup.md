# Gmail API Setup Guide

This guide will help you set up the Gmail API integration for the RemarkableAI application.

## Prerequisites

1. A Google Cloud Platform account
2. A Gmail account that you want to use with the application

## Step 1: Create a Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the project dropdown at the top of the page
3. Click "New Project"
4. Enter a name for your project (e.g., "RemarkableAI")
5. Click "Create"

## Step 2: Enable the Gmail API

1. In your new project, go to "APIs & Services" > "Library"
2. Search for "Gmail API"
3. Click on "Gmail API" in the results
4. Click "Enable"

## Step 3: Configure OAuth Consent Screen

1. Go to "APIs & Services" > "OAuth consent screen"
2. Select "External" user type (unless you have a Google Workspace account)
3. Click "Create"
4. Fill in the required information:
   - App name: "RemarkableAI"
   - User support email: Your email address
   - Developer contact information: Your email address
5. Click "Save and Continue"
6. Under "Scopes", add the following scope:
   - `https://www.googleapis.com/auth/gmail.readonly`
7. Click "Save and Continue"
8. Add your email address as a test user
9. Click "Save and Continue"

## Step 4: Create OAuth Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. Choose "Desktop app" as the application type
4. Enter a name for your OAuth client (e.g., "RemarkableAI Desktop")
5. Click "Create"
6. Download the credentials file
7. Rename the downloaded file to `credentials.json`
8. Place `credentials.json` in the root directory of your project

## Step 5: Run the Setup Script

1. Make sure you have all the required Python packages installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the setup script:
   ```bash
   python scripts/setup_gmail.py
   ```

3. A browser window will open asking you to sign in to your Google account
4. Grant the requested permissions
5. The script will create a `token.pickle` file in your project directory

## Troubleshooting

If you encounter the "Access blocked" error:

1. Make sure you've added your email as a test user in the OAuth consent screen
2. Verify that you're using the correct credentials.json file
3. Check that the Gmail API is enabled in your project
4. Ensure you're using the correct scopes in your application
5. Try deleting the token.pickle file and running the setup script again

## Security Notes

- Keep your `credentials.json` and `token.pickle` files secure
- Do not commit these files to version control
- If you need to revoke access, go to your Google Account settings > Security > Third-party apps with account access 