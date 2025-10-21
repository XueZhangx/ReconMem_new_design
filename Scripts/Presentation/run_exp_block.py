# Import necessary libraries
from psychopy import visual, core, event  # , sound
import os, sys, configparser
import pandas as pd
import numpy as np
from datetime import datetime
import random

# Read config
exp_config = configparser.ConfigParser()
exp_config.read('config')
sub = int(sys.argv[1])
block = int(sys.argv[2])

exp_dir = exp_config['DIR']['exp_dir']
stimuli_dir = exp_config['DIR']['stimuli_dir']
font_name = exp_config['EXP']['font_name']
curr_sub_dir = os.path.join(exp_dir, 'sub-%02d' % sub)

# 修改：读取study和test两个文件
study_block_dir = os.path.join(curr_sub_dir, 'block%02d_study.csv' % block)
test_block_dir = os.path.join(curr_sub_dir, 'block%02d_test.csv' % block)

# 时间参数
stimulus_duration = float(exp_config['EXP']['stimulus_duration'])  # study图片呈现时间
stimulus_blank = float(exp_config['EXP']['stimulus_blank'])  # study空白间隔
study_question_duration = float(exp_config['EXP'].get('study_question_duration', '3'))  # study问题呈现时间
test_stimulus_duration = float(exp_config['EXP'].get('test_stimulus_duration', '1'))  # test图片呈现时间
test_question1_duration = float(exp_config['EXP'].get('test_question1_duration', '3'))  # test问题1呈现时间
test_question2_duration = float(exp_config['EXP'].get('test_question2_duration', '5'))  # test问题2呈现时间
break_duration = float(exp_config['EXP'].get('break_duration', '5'))  # study-test间隔

key_list = exp_config['EXP']['key_list'].split(', ')
confidence_keys = exp_config['EXP']['confidence_keys'].split(', ')
pre_exp_time = float(exp_config['EXP']['pre_exp_time'])
post_exp_time = float(exp_config['EXP']['post_exp_time'])
exp_mode = bool(exp_config['EXP']['exp_mode'])
print(exp_mode)
test_screen_size = (int(exp_config['EXP']['test_screen_width']), int(exp_config['EXP']['test_screen_height']))
exp_screen_size = (int(exp_config['EXP']['exp_screen_width']), int(exp_config['EXP']['exp_screen_height']))

# 加载数据
study_df = pd.read_csv(study_block_dir)
test_df = pd.read_csv(test_block_dir)

# 创建日志文件
timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
ses_info = 'Subject: {} \n Block: {} \n'.format(sub, block)
log_file = open(os.path.join(curr_sub_dir,
                             'experiment_log_block%02d_%s.txt' % (block, timestamp)), 'a')
# 记录会话信息
date_line = 'Session Start Time: {}\n'.format(datetime.now())
log_file.write(date_line)
log_file.write(ses_info)

# 创建实验窗口
if not exp_mode:
    win = visual.Window(size=test_screen_size, fullscr=exp_mode, color='gray', screen=1)
    print(win.getActualFrameRate())
    screen_size = test_screen_size
else:
    win = visual.Window(size=exp_screen_size, fullscr=exp_mode, color='gray', screen=1)
    print(win.getActualFrameRate())
    screen_size = exp_screen_size

# 创建 fixation dot
fixation_dot = visual.Circle(win, size=(0.02 * (screen_size[1] / screen_size[0]), 0.02),
                             fillColor='white', lineColor='white')
red_fixation_dot = visual.Circle(win, size=(0.02 * (screen_size[1] / screen_size[0]), 0.02),
                                 fillColor='red', lineColor='red')
fixation_cross = visual.TextStim(win, text="+", color='white')

# 创建时钟
stimulus_clock = core.Clock()


def check_button_preparation(phase_name, keys_to_check=None):
    """检查被试按键准备 - 依次检查每个按键"""
    if keys_to_check is None:
        keys_to_check = key_list

    for current_key in keys_to_check:
        key_text = visual.TextStim(win, text="{} 请按.".format(current_key), color='white', font=font_name)

        while True:
            # 检查所有按键，包括escape
            keys = event.getKeys(keyList=keys_to_check + ['escape'])

            # 如果按了escape键，退出程序
            if 'escape' in keys:
                win.close()
                core.quit()

            # 如果按了正确的键，跳出循环
            if current_key in keys:
                break

            # 显示提示文本
            key_text.draw()
            win.flip()

        win.flip()
        core.wait(0.2)  # 短暂间隔


# 倒计时函数
def countdown_timer(duration, message):
    """显示倒计时"""
    countdown_start = stimulus_clock.getTime()
    while stimulus_clock.getTime() - countdown_start < duration:
        count_down = int(np.ceil(duration - (stimulus_clock.getTime() - countdown_start)))
        instruction_text = visual.TextStim(win,
                                           text=f"{count_down}秒后\n{message}",
                                           color='white',
                                           font=font_name)
        instruction_text.draw()
        win.flip()



def run_study_part():
    """运行study部分"""
    print('Starting Study Part...')

    person_question = visual.TextStim(win, text="上一张图片中是否包含人物？\n有按1，没有按0",
                                      color='white', font=font_name)
    # 预加载图片
    study_stimuli = []
    for curr_row in study_df.iterrows():
        trial_data = curr_row[1]

        # 根据image_type判断图片类型
        if trial_data['image_type'] == 'filler':
            # filler图片从fillers_images目录加载
            img_dir = os.path.join(stimuli_dir, trial_data['Image_Index'])
        else:
            # target图片按原方式加载
            img_ind = trial_data['Image_Index']
            img_type = img_ind.split('-')[0]
            img_num = img_ind.split('-')[1]
            img_dir = os.path.join(stimuli_dir, '{}s'.format(img_type), '{}.jpg'.format(img_num))

        img_obj = visual.ImageStim(win, image=img_dir)
        study_stimuli.append(img_obj)

    # 初始化记录变量
    study_onset_times = np.ones((len(study_stimuli), 1)) * np.nan
    study_offset_times = np.ones((len(study_stimuli), 1)) * np.nan
    study_responses = np.ones((len(study_stimuli), 1)) * np.nan
    study_RTs = np.ones((len(study_stimuli), 1)) * np.nan
    study_question_positions = np.zeros((len(study_stimuli), 1))  # 0表示无问题，1表示有问题
    study_question_responses = np.ones((len(study_stimuli), 1)) * np.nan
    study_question_RTs = np.ones((len(study_stimuli), 1)) * np.nan

    # 呈现红色fixation dot表示开始
    red_fixation_dot.draw()
    win.flip()
    core.wait(0.2)

    for trial_count, (stimulus, trial_data) in enumerate(zip(study_stimuli, study_df.iterrows())):
        trial_info = trial_data[1]

        # 呈现图片
        img_startTime = stimulus_clock.getTime() - exp_startTime
        study_onset_times[trial_count] = img_startTime

        # 图片呈现循环
        stimulus.draw()
        win.flip()
        # TODO trigger（study_stim_on)

        picture_start = stimulus_clock.getTime()

        while stimulus_clock.getTime() - picture_start < stimulus_duration:
            keys = event.getKeys(keyList=key_list + ['escape'])
            if 'escape' in keys:
                win.close()
                core.quit()

        study_offset_times[trial_count] = stimulus_clock.getTime() - exp_startTime

        # 空白间隔
        fixation_dot.draw()
        win.flip()
        # TODO trigger（study_stim_off)
        core.wait(stimulus_blank)

        # 如果是filler图片，呈现人物检测问题
        if trial_info['has_question']:
            study_question_positions[trial_count] = 1
            question_start = stimulus_clock.getTime()
            response_made = False

            while stimulus_clock.getTime() - question_start < study_question_duration and not response_made:
                person_question.draw()
                win.flip()
                # TODO trigger（study_question_on)

                keys = event.getKeys(keyList=key_list + ['escape'], timeStamped=stimulus_clock)
                if 'escape' in [key[0] for key in keys]:
                    win.close()
                    core.quit()
                # TODO trigger（study_keypress)
                if keys and keys[0][0] != 'escape':
                    study_question_responses[trial_count] = keys[0][0]
                    study_question_RTs[trial_count] = keys[0][1] - question_start
                    response_made = True
                    log_file.write(
                        f'STUDY_FILLER_QUESTION, {block}, {trial_count}, {question_start}, {study_question_RTs[trial_count]}, {study_question_responses[trial_count]}\n')

            if not response_made:
                study_question_responses[trial_count] = 'NaN'
                study_question_RTs[trial_count] = 'NaN'

            # 问题后空白
            fixation_dot.draw()
            win.flip()
            core.wait(0.5)

    # 保存study数据
    study_df['Onset'] = study_onset_times
    study_df['Offset'] = study_offset_times
    study_df['Response'] = study_responses
    study_df['RT'] = study_RTs
    study_df['Question_Position'] = study_question_positions
    study_df['Question_Response'] = study_question_responses
    study_df['Question_RT'] = study_question_RTs

    study_output_dir = os.path.join(curr_sub_dir, 'study_response_block{}_{}.csv'.format(block, timestamp))
    study_df.to_csv(study_output_dir)

    return study_question_positions


def run_test_part():
    """运行test部分"""
    print('Starting Test Part...')

    # 预加载图片
    test_stimuli = []
    for curr_row in test_df.iterrows():
        img_ind = curr_row[1]['Image_Index']
        img_type = img_ind.split('-')[0]
        img_num = img_ind.split('-')[1]
        img_dir = os.path.join(stimuli_dir, '{}s'.format(img_type), '{}.jpg'.format(img_num))
        img_obj = visual.ImageStim(win, image=img_dir)
        test_stimuli.append(img_obj)

    # 加载自信度评分的卡通人物图片
    confidence_images = []
    for i in range(3):
        # 假设卡通图片在stimuli_dir的某个子目录中，比如'confidence'
        confidence_path = os.path.join(stimuli_dir, 'test_cr', '{}.png'.format(i))
        confidence_img = visual.ImageStim(win, image=confidence_path)
        confidence_images.append(confidence_img)

    # 创建问题文本
    question1_text = visual.TextStim(win, text="是否看过这张图片？\n看过按1，没看过按0",
                                     color='white', font=font_name)

    question2_text = visual.TextStim(win,
                                     text="你有多大自信做出刚才的判断？",
                                     color='white', pos=(0, 0.6), font=font_name)

    # 自信度选项文本
    confidence_labels = [
        visual.TextStim(win, text="←", color='white', pos=(-0.4, -0.8), font=font_name),
        visual.TextStim(win, text="↓", color='white', pos=(0, -0.8), font=font_name),
        visual.TextStim(win, text="→", color='white', pos=(0.4, -0.8), font=font_name),
    ]

    # 修改按键列表为方向键
    confidence_keys = ['left', 'down', 'right']  # 左箭头=不太确定，下箭头=有点确定，右箭头=非常确定

    # 初始化记录变量
    test_onset_times = np.ones((len(test_stimuli), 1)) * np.nan
    test_offset_times = np.ones((len(test_stimuli), 1)) * np.nan
    test_responses_q1 = np.ones((len(test_stimuli), 1)) * np.nan
    test_RTs_q1 = np.ones((len(test_stimuli), 1)) * np.nan
    test_responses_q2 = np.ones((len(test_stimuli), 1)) * np.nan
    test_RTs_q2 = np.ones((len(test_stimuli), 1)) * np.nan

    # 呈现红色fixation dot表示开始
    red_fixation_dot.draw()
    win.flip()
    core.wait(0.2)

    for trial_count, stimulus in enumerate(test_stimuli):
        # 呈现图片
        img_startTime = stimulus_clock.getTime() - exp_startTime
        test_onset_times[trial_count] = img_startTime

        stimulus.draw()
        win.flip()
        # TODO trigger（test_stim_on)
        core.wait(test_stimulus_duration)

        img_start = stimulus_clock.getTime()
        while stimulus_clock.getTime() - img_start < test_stimulus_duration:
            escape_keys = event.getKeys(keyList=['escape'])
            if 'escape' in escape_keys:
                win.close()
                core.quit()

        test_offset_times[trial_count] = stimulus_clock.getTime() - exp_startTime

        # 问题1：新旧判断
        question1_start = stimulus_clock.getTime()
        response_made = False

        while stimulus_clock.getTime() - question1_start < test_question1_duration and not response_made:
            question1_text.draw()
            win.flip()

            keys = event.getKeys(keyList=key_list, timeStamped=stimulus_clock)
            if keys:
                # TODO trigger（test_y/n_keypress)
                test_responses_q1[trial_count] = keys[0][0]
                test_RTs_q1[trial_count] = keys[0][1] - question1_start
                response_made = True
                log_file.write(
                    f'TEST_Q1, {block}, {trial_count}, {question1_start}, {test_RTs_q1[trial_count]}, {test_responses_q1[trial_count]}\n')

        if not response_made:
            test_responses_q1[trial_count] = 'NaN'
            test_RTs_q1[trial_count] = 'NaN'

        # 问题2：自信度判断
        question2_start = stimulus_clock.getTime()
        response_made = False

        # 在run_test_part函数中，修改问题2的按键检测部分：
        while stimulus_clock.getTime() - question2_start < test_question2_duration and not response_made:
            # 绘制问题文本（在屏幕最上方）
            question2_text.draw()

            # 绘制刚才的图片（位置在问题文本下方）
            stimulus.pos = (0, 0.2)  # 调整图片位置到中间偏上
            stimulus.draw()


            # 绘制三个卡通人物和标签
            for i, (img, label) in enumerate(zip(confidence_images, confidence_labels)):
                x_pos = -0.4 + i * 0.4  # 三个位置：左(-0.4)、中(0)、右(0.4)
                img.pos = (x_pos, -0.4)  # 图片位置在中间偏下

                img.draw()
                label.draw()

            win.flip()


            keys = event.getKeys(keyList=confidence_keys, timeStamped=stimulus_clock)  # 使用confidence_keys
            if keys:
                # TODO trigger（test_confidence_keypress)
                # 将方向键映射为数字
                key_mapping = {'left': '0', 'down': '1', 'right': '2'}
                test_responses_q2[trial_count] = key_mapping[keys[0][0]]
                test_RTs_q2[trial_count] = keys[0][1] - question2_start
                response_made = True
                log_file.write(
                    f'TEST_Q2, {block}, {trial_count}, {question2_start}, {test_RTs_q2[trial_count]}, {test_responses_q2[trial_count]}\n')

        if not response_made:
            test_responses_q2[trial_count] = 'NaN'
            test_RTs_q2[trial_count] = 'NaN'

        # trial间间隔
        fixation_dot.draw()
        win.flip()
        core.wait(0.5)

    # 保存test数据
    test_df['Onset'] = test_onset_times
    test_df['Offset'] = test_offset_times
    test_df['Response_Q1'] = test_responses_q1
    test_df['RT_Q1'] = test_RTs_q1
    test_df['Response_Q2'] = test_responses_q2
    test_df['RT_Q2'] = test_RTs_q2

    test_output_dir = os.path.join(curr_sub_dir, 'test_response_block{}_{}.csv'.format(block, timestamp))
    test_df.to_csv(test_output_dir)


# 主实验流程
# 1. 检查按键准备
check_button_preparation("study")

# 2. 显示study部分倒计时
countdown_timer(pre_exp_time, "Study部分开始")

# 3. 记录实验开始时间
exp_startTime = stimulus_clock.getTime()
line_info = 'Experiment started at {}\n'.format(exp_startTime)
log_file.write(line_info)

# 4. 运行study部分
question_positions = run_study_part()

# 5. study-test间隔
break_text = visual.TextStim(win, text="Study部分结束\n稍作休息，Test部分即将开始",
                             color='white', font=font_name)
break_text.draw()
win.flip()
core.wait(break_duration)

# 6. 检查test部分按键准备
check_button_preparation("test")
countdown_timer(pre_exp_time, "Test部分开始")

# 7. 运行test部分
run_test_part()

# 记录实验结束时间
exp_endTime = stimulus_clock.getTime()
log_file.write(f'Experiment ended at {exp_endTime}\n')
log_file.close()

# 结束倒计时
countdown_timer(post_exp_time, "实验结束")

# 清理并退出
win.close()
core.quit()