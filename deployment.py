import subprocess
import os

# streamlit app commands
streamlit_command = "streamlit run app.py"

# set the square developer credentials
os.environ['SQUARE_ACCESS_TOKEN"'] = "EAAAF7VAZ50eAjTDeyX2t2UBlizpsnLsNoW4GKu3zDHeU8eJzahFblqV00FyidHr"
os.environ['SQUARE_APP_ID']= "sq0idp-N6SaVXd0V2skinyDrOwVmw"
# env_variable ={
#     "SQUARE_ACCESS_TOKEN": square_token,
#     "SQUARE_APP_ID": square_app_id
# }


# # combine the environment variables
# command = f"{streamlit_command} --server.port=$PORT"
# for key, value in env_variable.items():
#     command = f"{command} --{key}={value}"

# Deploy the Streamlit app using subprocess
subprocess.call(streamlit_command, shell=True)
