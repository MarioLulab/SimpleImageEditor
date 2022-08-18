import cv2
import numpy as np
import sys
import os
import os.path as osp
from glob import glob

sys.path.append(".")
import utils

class ImageMeta():
    def __init__(self, file_name) -> None:
        self.file_name = file_name
        self.rot_angle = 0.
        self.scale = 1.
        self.transition = [0,0] # vertical, horizon

    def __str__(self) -> str:
        return "rot_angle={:5f}, scale={:5f}, transition={}".format(self.rot_angle, \
                                                                    self.scale, \
                                                                        self.transition)

class ImageView():
    def __init__(self, 
                 board_size=(2400, 1350),   # 16:9
                 img_size=(1920,1080)) :
        self.img_size = img_size
        self.last_windows_name = None

    def display_img(self, img_meta : ImageMeta, img : cv2.Mat):
        # display img on board ? or directly display ?
        # cv2.imshow(img_meta.file_name, img)
        if (self.last_windows_name is not None) and (self.last_windows_name != img_meta.file_name) :
            cv2.destroyWindow(self.last_windows_name)
        cv2.imshow(img_meta.file_name, img)
        self.last_windows_name = img_meta.file_name

        print(f"[{img_meta.file_name}] img meta info : {img_meta}")
        # cv2.imwrite("tmp.jpg", img)


class ImageMode():
    def __init__(self, img_dir) -> None:

        self.img_path = []
        self._count = 0

        assert type(img_dir) == str or type(img_dir) == list, "type(img_dir) must be list[str] or str"
        if type(img_dir) == str:
            img_dir = [img_dir]
        
        for dir in img_dir:
            self.img_path += sorted(glob(osp.join(dir, "*")))

    def __len__(self):
            return len(self.img_path)
        
    def read_image(self, filename : str):
        return cv2.imread(filename).astype(np.float32)
    
    def save_image(self, filename : str, img : cv2.Mat ):
        try :
            cv2.imwrite(filename, img)
            print(f"save_img to {filename}: Success")
        except:
            print(f"save_img to {filename}: Error")

    def save_image_and_annotation(self, img_meta : ImageMeta, save_root : str):
        pass


    def read_next(self):
        if self._count < len(self.img_path) - 1:
            ret = ImageMeta(self.img_path[self._count])
            self._count += 1
        else :
            raise Exception("It's the last frame. Can't read next")

        return ret
    
    def read_prev(self):
        if self._count >= 1:
            ret = ImageMeta(self.img_path[self._count])
            self._count -= 1
        else :
            raise Exception("It's the first frame. Can't read prev")

        return ret

class ImagePresenter():
    def __init__(self, img_mode : ImageMode, img_view : ImageView) -> None:
        self.current_frame = None
        self.current_frame_name = str()

        self.img_mode = img_mode
        self.img_view = img_view

    def rotate(self, img : cv2.Mat, angle : float):
        h,w = img.shape[0:2]
        center = np.array((h/2, w/2))
        scale = 1
        
        mat = utils.get_transform(center, scale, (h, w), angle)
        img = cv2.warpAffine(img, mat, (h, w)).astype(np.float32)
        return img

    def scale(self, img : cv2.Mat, scale_factor : float):
        h,w = img.shape[0:2]
        center = np.array((h/2, w/2))
        angle = 1

        mat = utils.get_transform(center, scale_factor, (h, w), angle)
        img = cv2.warpAffine(img, mat, (h, w)).astype(np.float32)
        return img

    def _get_transform_mat(self, img_meta : ImageMeta):
        rot_angle = img_meta.rot_angle
        scale = img_meta.scale
        

        h,w = self.current_frame.shape[0:2]
        center = np.array((h/2, w/2))

        mat = utils.get_transform(center, scale, (h,w), rot_angle)
        return mat

    def __call__(self, img_meta : ImageMeta):
        # if self.current_frame_name != img_meta.file_name:
        #     # open new frame
        #     self.current_frame_name = img_meta.file_name
        #     self.current_frame = self.img_mode.read_image(self.current_frame_name)

        self.current_frame_name = img_meta.file_name
        self.current_frame = self.img_mode.read_image(self.current_frame_name)

        # rotate and scale
        mat_1 = self._get_transform_mat(img_meta=img_meta)[:2]
    
        # transition
        mat_2 = np.float32([[1,0,img_meta.transition[1]], [0,1,img_meta.transition[0]]])

        h,w = self.current_frame.shape[0:2]

        self.current_frame = cv2.warpAffine(self.current_frame, mat_2, (w,h)).astype(np.float32)
        self.current_frame = cv2.warpAffine(self.current_frame, mat_1, (w,h)).astype(np.float32)


        self.img_view.display_img(img_meta, self.current_frame)


class Runner():
    def __init__(self, img_dir) :
        self.image_view = ImageView()
        self.image_mode = ImageMode(img_dir)
        self.image_presenter = ImagePresenter(self.image_mode, self.image_view)


    def run(self):
        
        # read image from folder
        img_meta = None

        while True:
            key = cv2.waitKey(0) & 0xFF
            # print(key)

            if img_meta == None:
                img_meta = self.image_mode.read_next()

            if key == ord('z'):
                # prev img
                img_meta = self.image_mode.read_prev()
            elif key == ord('c'):
                # next img
                img_meta = self.image_mode.read_next()                
            elif key == ord('a'):
                # transition left
                img_meta.transition[1] -= 5
            elif key == ord('d'):
                # transition right
                img_meta.transition[1] += 5
            elif key == ord('w'):
                # transition up
                img_meta.transition[0] -= 5
            elif key == ord('s'):
                # transition down
                img_meta.transition[0] += 5
            elif key == ord('e'):
                # clockwise rotate
                img_meta.rot_angle -= 3
            elif key == ord('q'):
                # anti-clockwise rotate
                img_meta.rot_angle += 3
            elif key == ord('j'):
                # scale smaller
                img_meta.scale -= 0.05
            elif key == ord('l'):
                # scale bigger
                img_meta.scale += 0.05
            elif key == ord('o'):
                self.image_mode.save_image("save_img.jpg", \
                                        self.image_presenter.current_frame)
            elif key == ord('p'):
                # exit
                break

            if img_meta is not None :
                self.image_presenter(img_meta)
                self.image_mode.save_image("tmp.jpg", self.image_presenter.current_frame)


if __name__ == "__main__":
    img_dirs = ["img/test", "img/train"]
    runner = Runner(img_dirs)
    runner.run()