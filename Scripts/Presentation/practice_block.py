# Import necessary libraries
from psychopy import visual, core, event
import os, sys, configparser
import pandas as pd
import numpy as np
from datetime import datetime
import random

# Read config
exp_config = configparser.ConfigParser()
exp_config.read('config')
sub = int(sys.argv[1])

exp_dir = exp_config['DIR']['exp_dir']
stimuli_dir = exp_config['DIR']['stimuli_dir']
font_name = exp_config['EXP']['font_name']
curr_sub_dir = os.path.join(exp_dir, 'sub-%02d' % sub)

# 时间参数 - 使用与主实验相同的参数
stimulus_duration = float(exp_config['EXP']['stimulus_duration'])  # study图片呈现时间
stimulus_blank = float(exp_config['EXP']['stimulus_blank'])  # study空白间隔
study_question_duration = float(exp_config['EXP'].get('study_question_duration', '5'))  # study问题呈现时间
test_stimulus_duration = float(exp_config['EXP'].get('test_stimulus_duration', '1'))  # test图片呈现时间
test_question1_duration = float(exp_config['EXP'].get('test_question1_duration', '5'))  # test问题1呈现时间
test_question2_duration = float(exp_config['EXP'].get('test_question2_duration', '5'))  # test问题2呈现时间
break_duration = float(exp_config['EXP'].get('break_duration', '5'))  # study-test间隔

key_list = exp_config['EXP']['key_list'].split(', ')
confidence_keys = exp_config['EXP'].get('confidence_keys', '0, 1, 2').split(', ')
pre_exp_time = float(exp_config['EXP']['pre_exp_time'])
post_exp_time = float(exp_config['EXP']['post_exp_time'])
exp_mode = bool(exp_config['EXP']['exp_mode'])
test_screen_size = (int(exp_config['EXP']['test_screen_width']), int(exp_config['EXP']['test_screen_height']))
exp_screen_size = (int(exp_config['EXP']['exp_screen_width']), int(exp_config['EXP']['exp_screen_height']))

# 加载filler图片
filler_img_info = pd.read_csv('../../Scripts/Preparation/filler_imgs_sEEG.csv')


# 创建练习block的刺激序列
def create_practice_sequence():
    """创建练习block的刺激序列"""
    # 随机选择30张filler图片
    practice_imgs = filler_img_info.sample(n=30, random_state=sub).reset_index(drop=True)

    # 前20张用于study，后10张用于test的新图片
    study_imgs = practice_imgs.iloc[:20]
    test_new_imgs = practice_imgs.iloc[20:]

    # 从study图片中随机选择10张作为test的旧图片
    test_old_imgs = study_imgs.sample(n=10, random_state=sub)

    # 创建study序列
    study_sequence = []
    study_img_order = study_imgs.sample(frac=1, random_state=sub).reset_index(drop=True)

    for i, (_, img_row) in enumerate(study_img_order.iterrows()):
        study_sequence.append({
            'trial_num': i + 1,
            'image_label': f'practice_S{i + 1}',
            'Image_Index': img_row['Index'],
            'type': 'study'
        })

    # 创建test序列（10张旧 + 10张新）
    test_old_sequence = []
    for i, (_, img_row) in enumerate(test_old_imgs.iterrows()):
        test_old_sequence.append({
            'image_label': f'practice_S{i + 1}',
            'Image_Index': img_row['Index'],
            'correct_response': 1,  # 旧图片正确反应是1
            'type': 'old'
        })

    test_new_sequence = []
    for i, (_, img_row) in enumerate(test_new_imgs.iterrows()):
        test_new_sequence.append({
            'image_label': f'practice_N{i + 1}',
            'Image_Index': img_row['Index'],
            'correct_response': 0,  # 新图片正确反应是0
            'type': 'new'
        })

    # 合并并打乱test序列
    test_sequence = test_old_sequence + test_new_sequence
    random.shuffle(test_sequence)

    # 添加trial编号
    for i, trial in enumerate(test_sequence):
        trial['trial_num'] = i + 1

    return study_sequence, test_sequence


# 创建日志文件
timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
ses_info = 'Subject: {} \n Block: Practice \n'.format(sub)
log_file = open(os.path.join(curr_sub_dir, 'practice_log_{}.txt'.format(timestamp)), 'a')
date_line = 'Session Start Time: {}\n'.format(datetime.now())
log_file.write(date_line)
log_file.write(ses_info)

# 创建实验窗口
if not exp_mode:
    win = visual.Window(size=test_screen_size, fullscr=exp_mode, color='gray', screen=1)
    screen_size = test_screen_size
else:
    win = visual.Window(size=exp_screen_size, fullscr=exp_mode, color='gray', screen=1)
    screen_size = exp_screen_size

# 创建 fixation dot
fixation_dot = visual.Circle(win, size=(0.02 * (screen_size[1] / screen_size[0]), 0.02),
                             fillColor='white', lineColor='white')
red_fixation_dot = visual.Circle(win, size=(0.02 * (screen_size[1] / screen_size[0]), 0.02),
                                 fillColor='red', lineColor='red')

# 创建时钟
stimulus_clock = core.Clock()


# 检查按键准备函数
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
        core.wait(0.2)


# 倒计时函数
def countdown_timer(duration, message):
    """显示倒计时"""
    countdown_start = stimulus_clock.getTime()
    while stimulus_clock.getTime() - countdown_start < duration:
        escape_keys = event.getKeys(keyList=['escape'])
        if 'escape' in escape_keys:
            win.close()
            core.quit()

        count_down = int(np.ceil(duration - (stimulus_clock.getTime() - countdown_start)))
        instruction_text = visual.TextStim(win,
                                           text=f"{count_down}秒后\n{message}",
                                           color='white',
                                           font=font_name)
        instruction_text.draw()
        win.flip()


# 运行study部分
def run_study_part(study_sequence):
    """运行study部分"""
    print('Starting Practice Study Part...')

    # 创建问题文本
    person_question = visual.TextStim(win, text="上一张图片中是否包含人物？\n有按1，没有按0",
                                     color='white',
                                     font=font_name)

    # 预加载图片
    study_stimuli = []
    for trial in study_sequence:
        img_ind = trial['Image_Index']
        img_type = img_ind.split('-')[0]
        img_num = img_ind.split('-')[1]
        img_dir = os.path.join(stimuli_dir, '{}s'.format(img_type), '{}.jpg'.format(img_num))
        img_obj = visual.ImageStim(win, image=img_dir)
        study_stimuli.append(img_obj)

    # 记录变量
    study_onset_times = np.ones((len(study_stimuli), 1)) * np.nan
    study_offset_times = np.ones((len(study_stimuli), 1)) * np.nan
    study_question_responses = np.ones((len(study_stimuli), 1)) * np.nan
    study_question_RTs = np.ones((len(study_stimuli), 1)) * np.nan

    # 呈现红色fixation dot表示开始
    red_fixation_dot.draw()
    win.flip()
    core.wait(0.2)

    for trial_count, (trial, stimulus) in enumerate(zip(study_sequence, study_stimuli)):
        # 呈现图片
        img_startTime = stimulus_clock.getTime()
        study_onset_times[trial_count] = img_startTime

        # 图片呈现循环
        stimulus.draw()
        win.flip()
        picture_start = stimulus_clock.getTime()

        while stimulus_clock.getTime() - picture_start < stimulus_duration:
            escape_keys = event.getKeys(keyList=['escape'])
            if 'escape' in escape_keys:
                win.close()
                core.quit()

        study_offset_times[trial_count] = stimulus_clock.getTime()

        # 空白间隔
        fixation_dot.draw()
        win.flip()
        blank_start = stimulus_clock.getTime()
        while stimulus_clock.getTime() - blank_start < stimulus_blank:
            escape_keys = event.getKeys(keyList=['escape'])
            if 'escape' in escape_keys:
                win.close()
                core.quit()

        # 在study部分随机选择一些trial插入人物检测问题
        # 这里我们选择20%的trial（4个）来呈现问题
        if trial_count % 5 == 0:  # 每5个trial呈现一个问题
            question_start = stimulus_clock.getTime()
            response_made = False

            while stimulus_clock.getTime() - question_start < study_question_duration and not response_made:
                person_question.draw()
                win.flip()

                keys = event.getKeys(keyList=key_list + ['escape'], timeStamped=stimulus_clock)
                if 'escape' in [key[0] for key in keys]:
                    win.close()
                    core.quit()

                if keys and keys[0][0] != 'escape':
                    study_question_responses[trial_count] = keys[0][0]
                    study_question_RTs[trial_count] = keys[0][1] - question_start
                    response_made = True
                    log_file.write(f'STUDY_QUESTION, practice, {trial_count}, {question_start}, {study_question_RTs[trial_count]}, {study_question_responses[trial_count]}\n')

            if not response_made:
                study_question_responses[trial_count] = 'NaN'
                study_question_RTs[trial_count] = 'NaN'

            # 问题后空白
            fixation_dot.draw()
            win.flip()
            core.wait(0.5)

    # 保存study数据
    study_df = pd.DataFrame(study_sequence)
    study_df['Onset'] = study_onset_times
    study_df['Offset'] = study_offset_times
    study_df['Question_Response'] = study_question_responses
    study_df['Question_RT'] = study_question_RTs

    return study_df


# 运行test部分
def run_test_part(test_sequence):
    """运行test部分"""
    print('Starting Practice Test Part...')

    # 预加载图片
    test_stimuli = []
    for trial in test_sequence:
        img_ind = trial['Image_Index']
        img_type = img_ind.split('-')[0]
        img_num = img_ind.split('-')[1]
        img_dir = os.path.join(stimuli_dir, '{}s'.format(img_type), '{}.jpg'.format(img_num))
        img_obj = visual.ImageStim(win, image=img_dir)
        test_stimuli.append(img_obj)

    # 加载自信度评分的卡通人物图片
    confidence_images = []
    for i in range(3):
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
    confidence_keys = ['left', 'down', 'right']

    # 记录变量（保持不变）
    test_onset_times = np.ones((len(test_stimuli), 1)) * np.nan
    test_offset_times = np.ones((len(test_stimuli), 1)) * np.nan
    test_responses_q1 = np.ones((len(test_stimuli), 1)) * np.nan
    test_RTs_q1 = np.ones((len(test_stimuli), 1)) * np.nan
    test_responses_q2 = np.ones((len(test_stimuli), 1)) * np.nan
    test_RTs_q2 = np.ones((len(test_stimuli), 1)) * np.nan
    test_accuracy = np.ones((len(test_stimuli), 1)) * np.nan

    # 呈现红色fixation dot表示开始
    red_fixation_dot.draw()
    win.flip()
    core.wait(0.2)

    for trial_count, (trial, stimulus) in enumerate(zip(test_sequence, test_stimuli)):
        # 呈现图片
        img_startTime = stimulus_clock.getTime()
        test_onset_times[trial_count] = img_startTime

        stimulus.draw()
        win.flip()
        core.wait(test_stimulus_duration)

        test_offset_times[trial_count] = stimulus_clock.getTime()

        # 问题1：新旧判断
        question1_start = stimulus_clock.getTime()
        response_made = False

        while stimulus_clock.getTime() - question1_start < test_question1_duration and not response_made:
            question1_text.draw()
            win.flip()

            keys = event.getKeys(keyList=key_list + ['escape'], timeStamped=stimulus_clock)
            if 'escape' in [key[0] for key in keys]:
                win.close()
                core.quit()

            if keys and keys[0][0] != 'escape':
                test_responses_q1[trial_count] = keys[0][0]
                test_RTs_q1[trial_count] = keys[0][1] - question1_start
                response_made = True

                if test_responses_q1[trial_count] == trial['correct_response']:
                    test_accuracy[trial_count] = 1
                else:
                    test_accuracy[trial_count] = 0

                log_file.write(
                    f'TEST_Q1, practice, {trial_count}, {question1_start}, {test_RTs_q1[trial_count]}, {test_responses_q1[trial_count]}\n')

        if not response_made:
            test_responses_q1[trial_count] = 'NaN'
            test_RTs_q1[trial_count] = 'NaN'
            test_accuracy[trial_count] = 'NaN'

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
                # 将方向键映射为数字
                key_mapping = {'left': '0', 'down': '1', 'right': '2'}
                test_responses_q2[trial_count] = key_mapping[keys[0][0]]
                test_RTs_q2[trial_count] = keys[0][1] - question2_start
                response_made = True
                log_file.write(
                    f'TEST_Q2, practice, {trial_count}, {question2_start}, {test_RTs_q2[trial_count]}, {test_responses_q2[trial_count]}\n')

        if not response_made:
            test_responses_q2[trial_count] = 'NaN'
            test_RTs_q2[trial_count] = 'NaN'

        # 在问题2回答之后，trial间间隔之前添加反馈：

        if not response_made:
            test_responses_q2[trial_count] = 'NaN'
            test_RTs_q2[trial_count] = 'NaN'

        # 添加对问题1回答正确与否的反馈
        if test_responses_q1[trial_count] != 'NaN':  # 确保问题1有回答
            if test_accuracy[trial_count] == 1:
                feedback_text = visual.TextStim(win, text="正确", color='blue', pos=(0, 0), font=font_name, height=0.1)
            else:
                feedback_text = visual.TextStim(win, text="错误", color='red', pos=(0, 0), font=font_name, height=0.1)

            # 显示反馈1秒
            feedback_text.draw()
            win.flip()
            core.wait(1.0)

        # trial间间隔
        fixation_dot.draw()
        win.flip()
        core.wait(0.5)

    # 保存test数据
    test_df = pd.DataFrame(test_sequence)
    test_df['Onset'] = test_onset_times
    test_df['Offset'] = test_offset_times
    test_df['Response_Q1'] = test_responses_q1
    test_df['RT_Q1'] = test_RTs_q1
    test_df['Response_Q2'] = test_responses_q2
    test_df['RT_Q2'] = test_RTs_q2
    test_df['Accuracy'] = test_accuracy

    return test_df





# 主实验流程
# 1. 创建练习序列
study_sequence, test_sequence = create_practice_sequence()

# 2. 检查study部分按键准备
check_button_preparation("练习study部分", key_list)

# 3. 显示study部分倒计时
countdown_timer(pre_exp_time, "练习Study部分")

# 4. 记录实验开始时间
exp_startTime = stimulus_clock.getTime()
line_info = 'Practice started at {}\n'.format(exp_startTime)
log_file.write(line_info)

# 5. 运行study部分
study_df = run_study_part(study_sequence)

# 6. study-test间隔
break_text = visual.TextStim(win, text="Study部分结束\n稍作休息，Test部分即将开始",
                             color='white',
                             font=font_name)
break_text.draw()
win.flip()
core.wait(break_duration)

# 7. 检查test部分按键准备
check_button_preparation("练习test部分", key_list)

# 8. 运行test部分
test_df = run_test_part(test_sequence)

# 保存数据
study_output_dir = os.path.join(curr_sub_dir, 'practice_study_{}.csv'.format(timestamp))
test_output_dir = os.path.join(curr_sub_dir, 'practice_test_{}.csv'.format(timestamp))
study_df.to_csv(study_output_dir, index=False)
test_df.to_csv(test_output_dir, index=False)

# 记录实验结束时间
exp_endTime = stimulus_clock.getTime()
log_file.write(f'Practice ended at {exp_endTime}\n')
log_file.close()

# 结束倒计时
countdown_timer(post_exp_time, "练习结束")

# 清理并退出
win.close()
core.quit()