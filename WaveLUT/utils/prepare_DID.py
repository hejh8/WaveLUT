import os
import os.path as osp
import random
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import glob_file_list

def sep_DID(DID_data_path, save_path):
    """Separate SDSD dataset.

        Args:
            DID_data_path (str): Path to SDSD DID part.
            outdoor_data_path (str): Path to SDSD outdoor part.
            save_path (str): Path to save dataset.
    """
    if not os.path.isdir(DID_data_path):
        print('Error: No source DID_data found')
        exit(0)


    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    DID_test =['video397', 'video200', 'video93', 'video322', 'video403', 'video228', 'video261', 'video314', 'video271', 'video328', 
                  'video288', 'video19', 'video327', 'video337', 'video92', 'video90', 'video120', 'video60', 'video305', 'video51', 
                  'video309', 'video300', 'video77', 'video394', 'video409', 'video396', 'video59', 'video102', 'video400', 'video344', 
                  'video391', 'video240', 'video243', 'video315', 'video302', 'video319', 'video269', 'video332', 'video108', 'video41', 
                  'video410', 'video356', 'video212', 'video237', 'video34', 'video303', 'video193', 'video345', 'video194', 'video383', 
                  'video159', 'video88', 'video24', 'video323', 'video353', 'video44', 'video333', 'video16', 'video95', 'video405', 
                  'video361', 'video14', 'video247', 'video376', 'video129', 'video219', 'video317', 'video23', 'video9', 'video313', 
                  'video399', 'video316', 'video217', 'video264', 'video40', 'video387', 'video178', 'video67', 'video284', 'video360', 
                  'video208', 'video389', 'video27', 'video251', 'video97', 'video18','video55', ]
    

    if not osp.exists(save_path + '/DID_np'):
        os.mkdir(save_path + '/DID_np')
        os.mkdir(save_path + '/DID_np/train')
        os.mkdir(save_path + '/DID_np/train/input')
        os.mkdir(save_path + '/DID_np/train/GT')
        os.mkdir(save_path + '/DID_np/test')
        os.mkdir(save_path + '/DID_np/test/input')
        os.mkdir(save_path + '/DID_np/test/GT')

    DID_data_LQ_path = DID_data_path + '/input'
    DID_data_GT_path = DID_data_path + '/GT'

    #DID_np/input/pairX
    DID_data_LQ_pairs_path = glob_file_list(DID_data_LQ_path)
    DID_data_GT_pairs_path = glob_file_list(DID_data_GT_path)

    DID_videos_LQ = []
    DID_videos_GT = []
    DID_videos_test_LQ = {}
    DID_videos_test_GT = {}


    for DID_pairX_LQ, DID_pairX_GT in zip(DID_data_LQ_pairs_path, DID_data_GT_pairs_path):
        pair_name = osp.basename(DID_pairX_GT)
        if pair_name in DID_test:
            video_LQ = glob_file_list(DID_pairX_LQ)
            video_GT = glob_file_list(DID_pairX_GT)
            video_LQ = video_LQ[0:30]
            video_GT = video_GT[0:30]
            frames_number = len(video_GT)
            if frames_number % 7 == 0:
                count = 0
                for shortvideo_start_index in range(0, frames_number, 7):
                    count += 1
                    shortvideo_end_index = shortvideo_start_index + 7
                    shortvideo_LQ = []
                    shortvideo_GT = []
                    for shortvideo_frame_index in range(shortvideo_start_index, shortvideo_end_index):
                        shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                        shortvideo_GT.append(video_GT[shortvideo_frame_index])
                    DID_videos_test_LQ[pair_name + '_' + str(count)] = shortvideo_LQ
                    DID_videos_test_GT[pair_name + '_' + str(count)] = shortvideo_GT
            else:
                frames_number_7 = int(frames_number / 7)
                lastshortvideo_start_index = frames_number_7 * 7 - (7 - frames_number % 7)
                count = 0
                for shortvideo_start_index in range(0, frames_number_7 * 7, 7):
                    count += 1
                    shortvideo_end_index = shortvideo_start_index + 7
                    shortvideo_LQ = []
                    shortvideo_GT = []
                    for shortvideo_frame_index in range(shortvideo_start_index, shortvideo_end_index):
                        shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                        shortvideo_GT.append(video_GT[shortvideo_frame_index])
                    DID_videos_test_LQ[pair_name + '_' + str(count)] = shortvideo_LQ
                    DID_videos_test_GT[pair_name + '_' + str(count)] = shortvideo_GT

                shortvideo_LQ = []
                shortvideo_GT = []
                count += 1
                for shortvideo_frame_index in range(lastshortvideo_start_index, frames_number):
                    shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                    shortvideo_GT.append(video_GT[shortvideo_frame_index])
                DID_videos_test_LQ[pair_name + '_' + str(count)] = shortvideo_LQ
                DID_videos_test_GT[pair_name + '_' + str(count)] = shortvideo_GT

            continue

        video_LQ = glob_file_list(DID_pairX_LQ)
        video_GT = glob_file_list(DID_pairX_GT)
        frames_number = len(video_GT)
        if frames_number % 7 == 0:
            for shortvideo_start_index in range(0, frames_number, 7):
                shortvideo_end_index = shortvideo_start_index + 7
                shortvideo_LQ = []
                shortvideo_GT = []
                for shortvideo_frame_index in range(shortvideo_start_index, shortvideo_end_index):
                    shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                    shortvideo_GT.append(video_GT[shortvideo_frame_index])
                DID_videos_LQ.append(shortvideo_LQ)
                DID_videos_GT.append(shortvideo_GT)
        else:
            frames_number_7 = int(frames_number / 7)
            lastshortvideo_start_index = frames_number_7 * 7 - (7 - frames_number % 7)
            for shortvideo_start_index in range(0, frames_number_7 * 7, 7):
                shortvideo_end_index = shortvideo_start_index + 7
                shortvideo_LQ = []
                shortvideo_GT = []
                for shortvideo_frame_index in range(shortvideo_start_index, shortvideo_end_index):
                    shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                    shortvideo_GT.append(video_GT[shortvideo_frame_index])
                DID_videos_LQ.append(shortvideo_LQ)
                DID_videos_GT.append(shortvideo_GT)

            shortvideo_LQ = []
            shortvideo_GT = []
            for shortvideo_frame_index in range(lastshortvideo_start_index, frames_number):
                shortvideo_LQ.append(video_LQ[shortvideo_frame_index])
                shortvideo_GT.append(video_GT[shortvideo_frame_index])
            DID_videos_LQ.append(shortvideo_LQ)
            DID_videos_GT.append(shortvideo_GT)


    videos_number = len(DID_videos_GT)
    for pair_index in range(1, len(DID_videos_GT) + 1):
        save_path_LQ_pairX = save_path + '/DID_np/train/input/pair' + str(pair_index)
        save_path_GT_pairX = save_path + '/DID_np/train/GT/pair' + str(pair_index)

        if not osp.exists(save_path_LQ_pairX):
            os.mkdir(save_path_LQ_pairX)
        if not osp.exists(save_path_GT_pairX):
            os.mkdir(save_path_GT_pairX)

        choose_video_index = random.randint(0, videos_number-1)
        choose_shortvideo_LQ = DID_videos_LQ.pop(choose_video_index)
        choose_shortvideo_GT = DID_videos_GT.pop(choose_video_index)
        videos_number -= 1

        count = 1
        for src_dir_LQ, src_dir_GT in zip(choose_shortvideo_LQ, choose_shortvideo_GT):
            new_filename = '0' + str(count) + '.jpg'
            count += 1

            dst_dir_LQ = osp.join(save_path_LQ_pairX, new_filename)
            dst_dir_GT = osp.join(save_path_GT_pairX, new_filename)

            shutil.copyfile(src_dir_LQ, dst_dir_LQ)
            shutil.copyfile(src_dir_GT, dst_dir_GT)

    for k, v in DID_videos_test_LQ.items():
        save_path_LQ_pairX = save_path + '/DID_np/test/input/' + k

        if not osp.exists(save_path_LQ_pairX):
            os.mkdir(save_path_LQ_pairX)

        count = 1
        for src_dir_LQ in v:
            new_filename = '0' + str(count) + '.jpg'
            count += 1

            dst_dir_LQ = osp.join(save_path_LQ_pairX, new_filename)
            shutil.copyfile(src_dir_LQ, dst_dir_LQ)

    for k, v in DID_videos_test_GT.items():
        save_path_GT_pairX = save_path + '/DID_np/test/GT/' + k

        if not osp.exists(save_path_GT_pairX):
            os.mkdir(save_path_GT_pairX)

        count = 1
        for src_dir_GT in v:
            new_filename = '0' + str(count) + '.jpg'
            count += 1

            dst_dir_GT = osp.join(save_path_GT_pairX, new_filename)
            shutil.copyfile(src_dir_GT, dst_dir_GT)


def main():
    sep_DID('', '')

if __name__ == "__main__":
    main()

