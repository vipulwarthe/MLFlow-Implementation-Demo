# MLFlow Implementation

Demo Link:
    
    https://chatgpt.com/c/68c120cc-354c-8333-88dc-eeefed2d57bf 
    Repo link: https://github.com/vipulwarthe/MLFlow-Implementation-Demo

Launch EC2 ubuntu 22.04/t2.medium/all-traffic or 22/443/5000 port need to open / 20gb storage 

1) SSH to instance & update
   
       ssh -i /path/to/key.pem ubuntu@<EC2_PUBLIC_IP>
       sudo apt update && sudo apt upgrade -y

 2) Install Python, venv, git, awscli (optional)

        sudo apt install -y python3 python3-pip python3-venv git
        sudo apt install -y awscli       # optional but handy:

(If you plan to use S3 later, awscli helps test S3. If you use an instance IAM role you don't need credentials on the box.)

3) Create project dir & virtualenv:

       mkdir ~/mlflow-server && cd ~/mlflow-server
       python3 -m venv venv
       source venv/bin/activate
       pip install --upgrade pip

4) Install MLflow (and sklearn for demo)

       pip install mlflow==3.3.2 scikit-learn boto3              # boto3 only if you'll use S3 artifacts later

5) Start MLflow Tracking Server (local backend + local artifacts)

       nohup mlflow server --backend-store-uri sqlite:///$(pwd)/mlflow.db \
         --default-artifact-root $(pwd)/mlruns --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &

6) Open Security Group to your IP on port 5000 (or use nginx/80)

   Then browse: http://<EC2_PUBLIC_IP>:5000 → you should see MLflow UI. (If you used nginx, use port 80.)

7) Create a tiny demo training script app.py
   vi app.py  and add above python experiment script

8) Point your client to remote MLflow server and run the script

       export MLFLOW_TRACKING_URI=http://<EC2_PUBLIC_IP>:5000           
       python app.py

   You should see a new run appear in the MLflow UI.     http://<EC2_PUBLIC_IP>:5000 
   
