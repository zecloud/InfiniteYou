import json
import os
import random
from io import BytesIO
from cachetools import cached
from dapr.clients import DaprClient
from dapr.ext.grpc import App, BindingRequest
from huggingface_hub import snapshot_download,hf_hub_download
import logging
#from cloudevents.sdk.event import v1
#from dapr.clients.grpc._response import TopicEventResponse
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import traceback
import torch
from huggingface_hub import snapshot_download
from pillow_heif import register_heif_opener

from pipelines.pipeline_infu_flux import InfUFluxPipeline

logging.basicConfig(level = logging.INFO)
app = App()
ACCOUNT_NAME = os.getenv("ACCOUNT_NAME", "fluxstorageaca")
account_url="https://"+ACCOUNT_NAME+".blob.core.windows.net"
CONTAINER_NAME = os.getenv("CONTAINER_NAME","ltxavatarjob")

# Register HEIF support for Pillow
register_heif_opener()

class ModelVersion:
    STAGE_1 = "sim_stage1"
    STAGE_2 = "aes_stage2"

    DEFAULT_VERSION = STAGE_1

def dlmodels():
    
    try:
        snapshot_download(repo_id='ByteDance/InfiniteYou', local_dir='./models/InfiniteYou', local_dir_use_symlinks=False)
    
        snapshot_download(repo_id='enhanceaiteam/Mystic', local_dir='./models/Mystic', local_dir_use_symlinks=False)
    except Exception as e:
        logging.error(f"Error downloading: {e}")
        logging.error("Traceback info:\n%s", traceback.format_exc())
        print(e)
       
@cached(cache={})
def initstabledif(model_version=ModelVersion.DEFAULT_VERSION):    
    dlmodels()
    model_path = f'./models/InfiniteYou/infu_flux_v1.0/{model_version}'
    print(f'loading model from {model_path}')
    pipeline = InfUFluxPipeline(
            base_model_path='./models/Mystic',
            infu_model_path=model_path,
            insightface_root_path='./models/InfiniteYou/supports/insightface',
            image_proj_num_tokens=8,
            infu_flux_version='v1.0',
            model_version=model_version,
        )
    return pipeline



def genimg(params):
    pipeline=initstabledif()
    prompt=params["prompt"]
    folder=params["folder"]
    idsave=params["idsave"]
    width=params.get("width",1280)
    height=params.get("height",720)
    data_bytes=runstabledif(pipeline,prompt,width,height)
    publish_and_save(data_bytes,folder,idsave)

def runstabledif(pipeline,prompt,width,height,input_image,guidance_scale=7.5,num_steps=16,infusenet_conditioning_scale=1.0,infusenet_guidance_start=0.0,infusenet_guidance_end=1.0,seed=None):
    if(not seed):
        seed = random.randint(0, 2 ** 32 - 1)

    try:
        image = pipeline(
            id_image=input_image,
            prompt=prompt,
            #control_image=control_image,
            seed=seed,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            infusenet_conditioning_scale=infusenet_conditioning_scale,
            infusenet_guidance_start=infusenet_guidance_start,
            infusenet_guidance_end=infusenet_guidance_end,
        )
        bIO = BytesIO()
        image.save(bIO, format="PNG")
        data_bytes=bIO.getvalue()
        return data_bytes
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        logging.error("Traceback info:\n%s", traceback.format_exc())
        print(e)

def azureupload(pathfile,data):
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=pathfile)
    blob_client.upload_blob(data)

def publish_and_save(img,folder,idsave):
    # Create a Dapr gRPC client
  with DaprClient() as client:
        outputfile=folder+"/"+idsave+".png"
        azureupload(outputfile,img)
        req_data = {'message':  outputfile} 
        # Create a typed message with content type and body
        resp = client.publish_event(
            pubsub_name='pubsubcomponent',
            topic_name=f'finishednarratorfluxjob',#/{idsave}
            data=json.dumps(req_data),
            data_content_type='application/json',
            publish_metadata={'rawPayload': 'true'},
        )
        logging.info('Published data: message sent')
        print('sent message and saved')

@app.binding('narratorfluxjobbinding')
def mytopic(request: BindingRequest):
    data = json.loads(request.text())
    print(f'Received: idsave={data["idsave"]}, prompt="{data["prompt"]}"' 
          ' content_type="{event.content_type}"',flush=True)
    try:
        genimg(data)
    except Exception as e:
        logging.error(f"Error processing message: {e}")
        logging.error("Traceback info:\n%s", traceback.format_exc())
        print(f"Error processing message: {e}",flush=True)

if __name__ == '__main__':
    app.register_health_check(lambda: print('Healthy'))
    app.run(50051)