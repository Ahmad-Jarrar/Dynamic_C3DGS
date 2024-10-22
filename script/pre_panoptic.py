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

def preparecolmappanoptic(folder, offset=0):
    # Only process folders inside `ims` and exclude `seg`
    folderlist = sorted(glob.glob(os.path.join(folder, "ims", "*/")))
    imagelist = []
    savedir = os.path.join(folder, "colmap_" + str(offset))
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    savedir = os.path.join(savedir, "input")
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    for camera_folder in folderlist:
        imagepath = os.path.join(camera_folder, f"{str(offset).zfill(6)}.jpg")  # Handle `.jpg`
        imagesavepath = os.path.join(savedir, camera_folder.split("/")[-2] + ".jpg")  # Save as `.jpg`

        if os.path.exists(imagepath):
            shutil.copy(imagepath, imagesavepath)
        else:
            print(f"Image {imagepath} not found. Skipping.")

def convertpanoptictocolmapdb(path, offset=0):
    originnumpy = os.path.join(path, "poses_bounds.npy")
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

    with open(originnumpy, 'rb') as numpy_file:
        poses_bounds = np.load(numpy_file)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)

        llffposes = poses.copy().transpose(1, 2, 0)
        w2c_matriclist = posetow2c_matrcs(llffposes)
        assert isinstance(w2c_matriclist, list)

        for i in range(len(poses)):
            cameraname = os.path.basename(camera_folders[i])[:-1]  # Remove trailing slash
            m = w2c_matriclist[i]
            colmapR = m[:3, :3]
            T = m[:3, 3]
            
            H, W, focal = poses[i, :, -1]
            
            colmapQ = rotmat2qvec(colmapR)

            imageid = str(i + 1)
            cameraid = imageid
            jpgname = cameraname + ".jpg"  # Handle `.jpg`
            
            line = imageid + " "
            line += " ".join(map(str, colmapQ)) + " "
            line += " ".join(map(str, T)) + " "
            line += cameraid + " " + jpgname + "\n"
            imagetxtlist.append(line)
            imagetxtlist.append("\n")

            focolength = focal
            model, width, height, params = i, W, H, np.array((focolength, focolength, W // 2, H // 2))

            camera_id = db.add_camera(1, width, height, params)
            cameraline = f"{i + 1} PINHOLE {width} {height} {focolength} {focolength} {W // 2} {H // 2}\n"
            cameratxtlist.append(cameraline)
            
            db.add_image(jpgname, camera_id, prior_q=colmapQ, prior_t=T, image_id=i + 1)
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
    print("Start preparing colmap image input")
    for offset in range(startframe, endframe):
        preparecolmappanoptic(os.path.join(folderpath, "ims"), offset)

    # Step 2: Prepare colmap database input
    print("Start preparing colmap database input")
    for offset in range(startframe, endframe):
        convertpanoptictocolmapdb(os.path.join(folderpath, "ims"), offset)

    # Step 3: Run colmap, per frame
    for offset in range(startframe, endframe):
        getcolmapsinglen3d(folderpath, offset)