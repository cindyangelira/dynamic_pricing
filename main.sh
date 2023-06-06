# main.sh

# Set the Cloud Function name and region
FUNCTION_NAME="my-streamlit-function"
REGION="us-central1"

# Deploy the Cloud Function
gcloud functions deploy $FUNCTION_NAME \
  --region=$REGION \
  --runtime python310 \
  --trigger-http \
  --allow-unauthenticated \
  --entry-point=main
