import os 
import glob 
import shutil
import numpy as np 
import sys 
import argparse
sys.path.append(".")
from thirdparty.gaussian_splatting.utils.my_utils import posetow2c_matrcs, rotmat2qvec
from thirdparty.colmap.pre_colmap import * 
from thirdparty.gaussian_splatting.helper3dg import getcolmapsinglen3d
import json

def preparecolmappanoptic(folder, offset=0):
    folderlist = sorted(glob.glob(os.path.join(folder, "ims", "*/")))
    savedir = os.path.join(folder, "colmap_" + str(offset), "input")
    os.makedirs(savedir, exist_ok=True)

    for camera_folder in folderlist:
        camera_id = camera_folder.split("/")[-2]  # Extract camera folder name (e.g., 0, 1, 2)
        imagepath = os.path.join(folder, "ims", camera_id, f"{str(offset).zfill(6)}.jpg")  # Correct path
        imagesavepath = os.path.join(savedir, camera_id + ".jpg")  # Save as `.jpg`

        if os.path.exists(imagepath):
            shutil.copy(imagepath, imagesavepath)
        else:
            print(f"Image {imagepath} not found. Skipping.")

def convertpanoptictocolmapdb(path, offset=0):
    # originnumpy = os.path.join(path, "poses_bounds.npy")
    train_meta = os.path.join(path, "train_meta.json")
    test_meta = os.path.join(path, "test_meta.json")


    camera_folders = sorted(glob.glob(os.path.join(path, 'ims', '*/')))  # Camera folders inside 'ims'
    projectfolder = os.path.join(path, "colmap_" + str(offset))
    manualfolder = os.path.join(projectfolder, "manual")

    if not os.path.exists(manualfolder):
        os.makedirs(manualfolder)

    savetxt = os.path.join(manualfolder, "images.txt")
    savecamera = os.path.join(manualfolder, "cameras.txt")
    savepoints = os.path.join(manualfolder, "points3D.txt")
    imagetxtlist = []
    cameratxtlist = []

    if os.path.exists(os.path.join(projectfolder, "input.db")):
        os.remove(os.path.join(projectfolder, "input.db"))

    db = COLMAPDatabase.connect(os.path.join(projectfolder, "input.db"))
    db.create_tables()

    camera_info = {}
    H, W = 0, 0

    with open(train_meta, "r") as f:
        train_meta = json.load(f)
        H, W = train_meta["h"], train_meta["w"]

        for cam_id, k, w2c in zip(train_meta["cam_id"][offset], train_meta["k"][offset], train_meta["w2c"][offset]):
            camera_info[cam_id] = (k, w2c)

    with open(test_meta, "r") as f:
        test_meta = json.load(f)

        for cam_id, k, w2c in zip(test_meta["cam_id"][offset], test_meta["k"][offset], test_meta["w2c"][offset]):
            camera_info[cam_id] = (np.array(k), np.array(w2c))
        
    for i in range(len(camera_folders)):
        print(f"Processing camera {i}")
        m = np.array(camera_info[i][1])
        colmapR = m[:3, :3]
        T = m[:3, 3]

        k = np.array(camera_info[i][0])
        
        colmapQ = rotmat2qvec(colmapR)

        imageid = str(i + 1)
        cameraid = imageid
        jpgname = imageid + ".jpg"  # Handle `.jpg`
        
        line = imageid + " "
        line += " ".join(map(str, colmapQ)) + " "
        line += " ".join(map(str, T)) + " "
        line += cameraid + " " + jpgname + "\n"
        imagetxtlist.append(line)
        imagetxtlist.append("\n")

        
        model, width, height, params = i, W, H, np.array((k[0, 0], k[1,1], k[0,2], k[1,2]))

        camera_id = db.add_camera(1, width, height, params, camera_id=i+1)
        cameraline = f"{i + 1} PINHOLE {width} {height} {k[0, 0]} {k[1,1]} {k[0,2]} {k[1,2]}\n"
        cameratxtlist.append(cameraline)
        
        print(f"Adding image {jpgname} with camera {camera_id+1} and image id {i+1}")
        db.add_image(jpgname, camera_id, prior_q=colmapQ, prior_t=T, image_id=i+1)
        db.commit()
        

    db.close()

    with open(savetxt, "w") as f:
        f.writelines(imagetxtlist)
    with open(savecamera, "w") as f:
        f.writelines(cameratxtlist)
    with open(savepoints, "w") as f:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folderpath", default="", type=str)
    parser.add_argument("--startframe", default=0, type=int)
    parser.add_argument("--endframe", default=150, type=int)

    args = parser.parse_args()
    folderpath = args.folderpath
    startframe = args.startframe
    endframe = args.endframe

    if startframe >= endframe:
        print("start frame must be smaller than end frame")
        quit()
    if startframe < 0 or endframe > 300:
        print("frame must be in range 0-300")
        quit()
    if not os.path.exists(folderpath):
        print("path does not exist")
        quit()
    
    if not folderpath.endswith("/"):
        folderpath = folderpath + "/"

    # Step 1: Prepare colmap input (skip frame extraction)
    # print("Start preparing colmap image input")
    # for offset in range(startframe, endframe):
    #     preparecolmappanoptic(folderpath, offset)

    # Step 2: Prepare colmap database input
    print("Start preparing colmap database input")
    for offset in range(startframe, endframe):
        print("Start preparing colmap database input for frame ", offset)
        convertpanoptictocolmapdb(folderpath, offset)

    # Step 3: Run colmap, per frame
    for offset in range(startframe, endframe):
        print("Start running colmap for frame ", offset)
        getcolmapsinglen3d(folderpath, offset)