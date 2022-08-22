import copy
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
        self.center = None
        self.shape = None   #(1920, 1080)

    def __str__(self) -> str:
        return "rot_angle={:5f}, scale={:5f}, transition={}, shape={}, center={}".format(
                                                                    self.rot_angle, \
                                                                    self.scale, \
                                                                    self.transition, \
                                                                    self.shape, \
                                                                    self.center)

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
    
    def __str__(self):
        return f"{self._count}/{len(self.img_path)}"
        
    def read_image(self, img_meta : ImageMeta):

        temp = cv2.imread(img_meta.file_name).astype(np.float32)
        h,w = temp.shape[:2]
        img_meta.shape = (h,w)
        # img_meta.center = [h/2 + img_meta.transition[0], w/2 + img_meta.transition[1]]
        img_meta.center = [h/2, w/2]
        return temp
    

    def save_image(self, filename : str, img : cv2.Mat ):
        try :
            cv2.imwrite(filename, img)
            print(f"save_img to {filename}: Success")
        except:
            print(f"save_img to {filename}: Error")

    def save_image_and_annotation(self, img_meta : ImageMeta, save_root : str):
        
        # file_name = img_meta.file_name
        # annt_name = file_name.replace("img_dir", "ann_dir")
        raise NotImplemented

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

        self.crop_location = [
            [np.array([0, 512]), np.array([1920, 512])],
            [np.array([0, 1080 - 512]), np.array([1920, 1080 - 512])],
        ]

    def _get_transform_mat(self, img_meta : ImageMeta):
        rot_angle = img_meta.rot_angle
        scale = img_meta.scale
        
        center = np.array(img_meta.center)

        mat = utils.get_transform(center, scale, img_meta.shape, rot_angle)
        return mat

    def transform(self, img_meta : ImageMeta, img : cv2.Mat):
        # rotate and scale
        mat_1 = self._get_transform_mat(img_meta=img_meta)[:2]
    
        # transition
        mat_2 = np.float32([[1,0,img_meta.transition[1]], [0,1,img_meta.transition[0]]])

        h,w = img.shape[0:2]

        img_transformed = cv2.warpAffine(img, mat_2, (w,h)).astype(np.float32)
        img_transformed = cv2.warpAffine(img_transformed, mat_1, (w,h)).astype(np.float32)
        # img_transformed = cv2.warpAffine(img_transformed, mat_2, (w,h)).astype(np.float32)

        return img_transformed

    def _generate_lined_frame(self):

        lined_frame = self.current_frame.copy()

        for point1, point2 in self.crop_location:
            lined_frame = cv2.line(lined_frame, point1, point2, color=(255,0,0), thickness=5)

        return lined_frame


    def __call__(self, img_meta : ImageMeta):

        self.current_frame_name = img_meta.file_name
        self.current_frame = self.img_mode.read_image(img_meta)

        self.current_frame = self.transform(img_meta, self.current_frame)

        lined_frame = self._generate_lined_frame()

        print(f"Processing Bar : {self.img_mode}")
        self.img_view.display_img(img_meta, lined_frame)
        self.img_mode.save_image("tmp.jpg", lined_frame)


def save_img_and_annt(img_meta : ImageMeta, image_presenter : ImagePresenter, image_mode : ImageMode, save_dir : str):
    '''
    save the preprocessed image and the relevant annotation image.
    e.x.:
        -- ttpla
            -- img_dir
                -- train
                    -- 0001.jpg
                    -- 0002.jpg
            -- ann_dir
                -- train
                    -- 0001.jpg
                    -- 0002.jpg

    transform both img('ttpla/img_dir/train/0001.jpg') and ann('ttpla/ann_dir/train/0001.jpg') using the same mat,
    and save them into '`save_dir`/img_dir/train/0001.jpg' and '`save_dir`/ann_dir/train/0001.jpg' repectively.
    '''

    ans = input("crop up, donw, or both?").strip()
    assert ans in ["up", "down", "both"], "You should input ['up', 'down', 'both']"


    # check dir structure
    if not osp.exists( osp.join(save_dir, 'img_dir') ):
        os.makedirs(osp.join(save_dir, 'img_dir', "train"))
        os.makedirs(osp.join(save_dir, 'img_dir', "test"))
        os.makedirs(osp.join(save_dir, 'img_dir', "val"))
    
    if not osp.exists( osp.join(save_dir, 'ann_dir') ):
        os.makedirs(osp.join(save_dir, 'ann_dir', 'train'))
        os.makedirs(osp.join(save_dir, 'ann_dir', 'test'))
        os.makedirs(osp.join(save_dir, 'ann_dir', 'val'))

    if not osp.exists( osp.join(save_dir, 'mask_dir') ):
        os.makedirs(osp.join(save_dir, 'mask_dir', 'train'))
        os.makedirs(osp.join(save_dir, 'mask_dir', 'test'))
        os.makedirs(osp.join(save_dir, 'mask_dir', 'val'))


    # get filename, annt_name, mask_name
    src_img_name = img_meta.file_name
    src_annt_name = img_meta.file_name.replace('img_dir', 'ann_dir')
    file_name = img_meta.file_name.split('/')[-1]
    dst_img_name = src_img_name.replace('ttpla', save_dir)
    dst_annt_name = src_annt_name.replace('ttpla', save_dir)
    dst_mask_name = src_img_name.replace('ttpla', save_dir).replace('img_dir', "mask_dir")


    # transform
    dst_img = image_presenter.current_frame
    
    annt_meta = copy.copy(img_meta)
    annt_meta.file_name = src_annt_name
    dst_annt = image_presenter.transform(annt_meta, image_mode.read_image(annt_meta))

    mask_meta = copy.copy(img_meta)
    mask_meta.file_name = dst_mask_name
    dst_mask = image_presenter.transform(mask_meta, np.ones_like(dst_img)*255)

    # crop
    dst_img_1_name = dst_img_name.replace('.', "_1.")
    dst_img_2_name = dst_img_name.replace('.', "_2.")

    # save
    if ans in ['up', 'both']:
        image_mode.save_image(dst_img_1_name, dst_img[:512, :, :])
    if ans in ['down', 'both']:    
        image_mode.save_image(dst_img_2_name, dst_img[1080-512:, :, :])

    # crop
    dst_annt_1_name = dst_annt_name.replace('.', "_1.")
    dst_annt_2_name = dst_annt_name.replace('.', "_2.")

    # save
    if ans in ['up', 'both']:
        image_mode.save_image(dst_annt_1_name, dst_annt[:512, :, :])
    if ans in ['down', 'both']:    
        image_mode.save_image(dst_annt_2_name, dst_annt[1080-512:, :, :])

    # crop
    dst_mask_1_name = dst_mask_name.replace('.', "_1.")
    dst_mask_2_name = dst_mask_name.replace('.', "_2.")

    # save
    if ans in ['up', 'both']:
        image_mode.save_image(dst_mask_1_name, dst_mask[:512, :, :])
    if ans in ['down', 'both']:    
        image_mode.save_image(dst_mask_2_name, dst_mask[1080-512:, :, :])

    # # save
    # image_mode.save_image(dst_img_name, dst_img)
    # image_mode.save_image(dst_annt_name, dst_annt)




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
                self.image_presenter(img_meta)
                save_img_and_annt(img_meta, self.image_presenter, self.image_mode, "output_dir")
            elif key == ord('p'):
                # exit
                break

            if img_meta is not None :
                self.image_presenter(img_meta)







if __name__ == "__main__":
    img_dirs = ["ttpla/img_dir/test", \
                "ttpla/img_dir/train", \
                "ttpla/img_dir/val"]    
    # img_dirs = ['img/train']

    runner = Runner(img_dirs)
    runner.run()