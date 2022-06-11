import uvicorn
from fastapi import FastAPI, File, UploadFile
from predict import prediction, get_model
import numpy as np
import cv2
import os
import wget
import zipfile

# data_url = "https://simon-storage.oss-cn-shanghai.aliyuncs.com/models.zip"
# if not os.path.exists('models'):
#     wget.download(data_url, out="models.zip")
#     zFile = zipfile.ZipFile("./models.zip", "r")
#
#     for fileM in zFile.namelist():
#         zFile.extract(fileM, "./")
#     zFile.close()
# 20_nodropout_LeNet5_mse_model_uncertainty_mse
# 20_nodropout_LeNet5_no_uncertainty_model
uncertainty_model = get_model("model_uncertainty_mse")
traditional_model = get_model("model")


app = FastAPI()

@app.get("/")
async def index():

    return {"info": {
        "name": "EDL vs TNN image classification backend",
        "author": "simon"
    }}


@app.post("/predict")
async def predict(img: UploadFile = File(...)):
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if not os.path.exists("images"):
        os.mkdir("images")
    filename = "images/target.jpg"
    cv2.imwrite(filename, img_color)
    # img_grey = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    uncertainty = prediction(uncertainty_model, filename, uncertainty=True)
    normal = prediction(traditional_model, filename, uncertainty=False)
    return {"uncertainty": uncertainty,
            "normal": normal}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
