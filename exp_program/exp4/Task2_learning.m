%% ==========================  1.1 Initializing  ==========================
% probabiliscitc feedback

clear all;

% % initialize recording device
IsRecording = 0;
if IsRecording
    fs = 44100;
    nbits = 16;
    nchannels = 1;
    recorder = audiorecorder(fs, nbits, nchannels);
end

taskID = 'Task2';

% input information
subID = input('Input the subject ID -> ', 's');
isPrac = input('Input the experiment mode -> ', 's');

subID = str2double(subID);
isPrac = str2double(isPrac);
behavior.subID = subID;
behavior.isPrac = isPrac;

% Create folders
folders = {fullfile(taskID,'Stimulus'), fullfile(taskID,'Data'), fullfile(taskID,'Recording')};  % folders for saving stimulus information, and behavior data
for i = 1:3
    if ~exist(folders{i}, 'dir')
        mkdir(folders{i});
    end
end

if isPrac
    sessionID = 0;
    dataname = fullfile(folders{2}, [taskID, '_', num2str(subID), '_prac_bhv']);
else
    sessionID = input('Input the session ID -> ', 's');
    sessionID = str2double(sessionID);
    behavior.sessionID = sessionID;
    dataname = fullfile(folders{2}, [taskID, '_', num2str(subID), '_', num2str(sessionID), '_bhv']);
end

% Create recording folders
if ~isPrac
    recfolder = fullfile(folders{3}, [num2str(subID),'_',num2str(sessionID)]);
else
    recfolder = fullfile(folders{3}, num2str(999));
end
if ~exist(recfolder, 'dir')
    mkdir(recfolder);
end

fid = fopen([dataname, '.csv'],'wt'); 
fprintf(fid, '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n', ...
    'condition','iSession','iBlock','iTrial','stiID','probCat1','probCat2', ...
    'choice','presskey','rating','feedback','choRT','rateRT','recDU'); % headers


%% =======================  1.2 Task organization  ========================

% set the unique random seed based on subID
s = RandStream('mt19937ar','Seed',subID*202603);
RandStream.setGlobalStream(s);

% task structure
Task2_settings = readtable('Task2_settings.csv'); % category structure, with format of [feature selection, feature space segmentation]


% maxBlock = [1, 5]; % maximum block number, separately for practice and formal task
% nTpB = [8, 64]; % trial number per block, separately for practice and formal task
% iToR = [nTpB(2)/4, nTpB(2)/2, nTpB(2)*3/4]; % rest when trial index

maxBlock = [1, 3]; 
nTpB = [8, 8]; 
iToR = 4;

if ~isPrac
    feature_names = Task2_settings.feature_names(Task2_settings.subID == subID);
    feature_names = string(split(feature_names, ',')');
    maxBlock = maxBlock(2);
    nTpB = nTpB(2);
else
    feature_names = ["green","yellow","pink","blue"];
    maxBlock = maxBlock(1);
    nTpB = nTpB(1);
end

% sampling of the first block
condition = 1; 
nCategory = 2;
nOption = 2;
trl_start = 1; % start trialID
trl_end = nTpB; % end trialID
sti_info = readtable(fullfile(folders{1}, 'Block1', [num2str(subID),'.csv'])); 
sti_info = sti_info(1:nTpB,:); % import stimulus information

% get the display order of stimulus
sti_order = Shuffle(trl_start:trl_end)';
sti_probcat1 = sti_info.prob_category1(sti_order);
sti_probcat2 = sti_info.prob_category2(sti_order);
sti_ambiguous = sti_info.ambiguous(sti_order);

% keys'number and category 
if isPrac || sessionID == 1
    keys_category = Shuffle(1:nOption);
else
    keys_category = load(fullfile(folders{2}, [taskID,'_',num2str(subID),'_',num2str(sessionID-1),'_bhv.mat'])).behavior.keys_category;
end


beta = 0.2; % parameter to control the generation for the next block
sigma = 0.3; % Gaussian width for stimulus sampling for the next block
beta = 10;
prototype1 = [0.25,0.5,0.5,0.5];
prototype2 = [0.75,0.5,0.5,0.5];
accstop = 0.9; % stop learning when accuracy meets criteria


% Feature image files (RGBA)
nFeature = 4;
img_feat_cell = cell(1, nFeature);
img_color = {'green','yellow','pink','blue'};
dot_color = [0, 100, 0; 255, 140, 0; 255, 20, 147; 0, 0, 139]';

for iFeature = 1:nFeature
    img_path = ['feature-', img_color{iFeature}, '.png'];
    [img_feat_rgb, ~, alpha] = imread(img_path, 'BackgroundColor', 'none');
    img_feat_cell{iFeature} = cat(3, img_feat_rgb, alpha);
end
[img_center_rgb, ~, alpha] = imread('center.png', 'BackgroundColor', 'none');
img_center = cat(3, img_center_rgb, alpha);

% stimulus feature range
feature_range = struct();
feature_range.body_length = 1;
feature_range.green_length = [feature_range.body_length, feature_range.body_length*4];
feature_range.yellow_length = [feature_range.body_length, feature_range.body_length*4];
feature_range.pink_length = [feature_range.body_length, feature_range.body_length*4];
feature_range.blue_length = [feature_range.body_length, feature_range.body_length*4];

angle = pi/4;
rotation_angles = [rad2deg(angle), -rad2deg(angle), rad2deg(pi)+rad2deg(angle), rad2deg(pi)-rad2deg(angle)];


behavior.condition = condition;
behavior.feature_names = feature_names;
behavior.sti_info = sti_info;
behavior.trl_start = trl_start;
behavior.trl_end = trl_end;
behavior.sti_order = sti_order;
behavior.sti_probcat1 = sti_probcat1;
behavior.sti_probcat2 = sti_probcat2;
behavior.keys_category = keys_category;
behavior.angle = angle;
behavior.rotation_angles = rotation_angles;
behavior.feature_range = feature_range;


%% ======================  1.3 Response key settings  =====================
KbName('UnifyKeyNames')

Keys = {'d','f','j','k'};
Key1 = KbName(Keys{1});
Key2 = KbName(Keys{2});
Key3 = KbName(Keys{3});
Key4 = KbName(Keys{4});
opt_names = [sprintf('%s',(upper(Keys{1}))),...
    sprintf('%s',(upper(Keys{2}))),...
    sprintf('%s',(upper(Keys{3}))),...
    sprintf('%s',(upper(Keys{4})))];

escape_key = KbName('ESCAPE');
space_key = KbName('SPACE');

% rating keys 
rating_key_names = {'1!','2@','3#','4$','5%','6^'};
rating_keys = cellfun(@KbName, rating_key_names);
scale = 1:6;

RestrictKeysForKbCheck([Key1, Key2, Key3, Key4, rating_keys, space_key, escape_key]);


%% ========================  1.4 Display settings  ========================

% select screen for display
screenID = 0;
w_rect = Screen('Rect', screenID); % the size of the window in display pixels
w_width = w_rect(3) - w_rect(1);
w_height = w_rect(4) - w_rect(2);
center_x = w_width / 2;
center_y = w_height / 2;

% calculate scaling factor transforming visual angle to display pixels (how
% many pixels correspond to 1 degree visual angle)
view_distance = 60; % distance between eye and screen (75 without eyetracker, 85~90 with eyetracker)
scr_width = 59.5; % width of screen (depend on the view distance)
scr_angle = round(2*180*atan(scr_width/(2*view_distance))/pi); % visual angle degree of the scr_width
deg2pix = round(w_width/scr_angle); % scaling factor

% transform fixation
font_size = 1 * deg2pix;
fix_radius = 0.003 * w_width;
fix_position = [center_x-fix_radius; center_y-fix_radius;...
    center_x+fix_radius; center_y+fix_radius];

% transform stimulus
body_visual_angle = 1; % visual angle of stimulus width
body_length_pix = round(body_visual_angle * deg2pix);
transform_to_pix_rate = body_length_pix / feature_range.body_length;

img_width_ratio = 1;

center_ratio = 8.5 / 5;
center_width = body_length_pix * center_ratio;
center_rect = [center_x-center_width/2, center_y-center_width/2, center_x+center_width/2, center_y+center_width/2];

dot_size = 8;

% 3) options
opt_radius = 25;
opt_width = 3;
opt_txt_size = 25;

% for 2 options
opt2_xys = [w_width*2/5-opt_radius, w_width*3/5-opt_radius; w_height*4/5-opt_radius, w_height*4/5-opt_radius;...
    w_width*2/5+opt_radius, w_width*3/5+opt_radius; w_height*4/5+opt_radius, w_height*4/5+opt_radius];
opt2_txt_xys = [w_width*2/5, w_height*4/5];
% for 4 options
opt4_xys = [w_width*2/7-opt_radius, w_width*3/7-opt_radius, w_width*4/7-opt_radius, w_width*5/7-opt_radius;
    w_height*4/5-opt_radius, w_height*4/5-opt_radius, w_height*4/5-opt_radius, w_height*4/5-opt_radius;
    w_width*2/7+opt_radius, w_width*3/7+opt_radius, w_width*4/7+opt_radius, w_width*5/7+opt_radius;
    w_height*4/5+opt_radius, w_height*4/5+opt_radius, w_height*4/5+opt_radius, w_height*4/5+opt_radius];
opt4_txt_xys = [w_width*2/7, w_height*4/5];

% ------ option settings based on condition
if condition == 1
    opt_xys = opt2_xys;
    opt_txt = opt_names(1,[2,3]);
    opt_txt_xys = opt2_txt_xys;
else
    opt_xys = opt4_xys;
    opt_txt = opt_names;
    opt_txt_xys = opt4_txt_xys;
end


% 4) microphone
mic_width = 0.1*w_width; % proportion of figure to the screen
ratio = 0.75; % width-height ratio of stimuli figure
mic_height = mic_width/ratio;
mic_rect = [center_x-mic_width/2, center_y-mic_height/2, center_x+mic_width/2, center_y+mic_height/2];


% basic colors
black_val = BlackIndex(screenID);
white_val = WhiteIndex(screenID);
grey_val = white_val / 2;
bg_color = grey_val;
fix_color = [204 204 204];
opt_color = black_val;
sopt_color = [242,242,242]; % higlighed selected option


behavior.deg2pix = deg2pix;
behavior.img_width_ratio = img_width_ratio;
behavior.center_ratio = center_ratio;
behavior.dot_color = dot_color;

%% ===================  1.5 Initialize a screen window  ===================

% use OpenGL for graphic rendering
AssertOpenGL;

% preference
Screen('Preference', 'SkipSyncTests', 0);
Screen('Preference', 'VBLTimestampingMode', 1);
Screen('Preference', 'ConserveVRAM', 8);
Screen('Preference', 'TextEncodingLocale', 'UTF8');
Screen('Preference', 'TextRenderer', 0);

% open window
[wPtr, ~] = PsychImaging('OpenWindow', screenID, bg_color, w_rect);

% set font face
Screen('TextFont', wPtr, 'simhei');

% set priority for matlab
priorityLevel = MaxPriority(wPtr);
Priority(priorityLevel);


% ------ microphone symbol
[microimg, ~, microalpha] = imread('microphone.png');
microimg(:, :, 2) = microimg(:, :, 1);
microimg(:, :, 3) = microimg(:, :, 1);
microimg(:, :, 4) = microalpha;
mic_texture = Screen('MakeTexture', wPtr, microimg);



%% =========================  1.6 Introductions  ==========================

welcome_txt = '任务2正式阶段\n\n准备好后请按【空格】键开始';
welcome_prc_txt = '任务2练习阶段\n\n准备好后请按【空格】键开始';
rec_txt = '请按【空格】键开始录音';
rest_txt = '休息一下，倒计时%i秒';
restend_txt = '休息好后请按【空格】键继续';
blockend_txt = '你在过去一段时间的正确率为%3.3f\n\n请按【空格】键继续';
end_txt = '任务2到此结束\n\n你的总正确率为%3.3f';
end_prc_txt = '练习结束\n\n请向主试示意';



%% ===================  1.7 Experiment timing settings  ===================

fix_numSecs = 1; 
fixj_numSecs = 0.3;
sopt_numSecs = 0.5;
prs_numSecs = 1.5;
ratemax_numSecs = 30;
% fdb_numSecs = 2;
blank_numSecs = 0.5;
rest_numSecs = 5;


% the vertical refresh rate of the monitor
ifi = Screen('GetFlipInterval', wPtr);

% fixation duration
fix_numFrames = round(fix_numSecs / ifi);
fix_fixed = fix_numFrames * ifi;

% fixation jitter
fixj_numFrames = round(fixj_numSecs / ifi);
fix_jitter = fixj_numFrames * ifi;

% selected option duration
sopt_numFrames = round(sopt_numSecs / ifi);
sopt_duration = sopt_numFrames * ifi;

% press duration to stop the recording
prs_numFrames = round(prs_numSecs / ifi);
prs_duration = prs_numFrames * ifi;

% rating max duration
ratemax_numFrames = round(ratemax_numSecs / ifi);
ratemax_duration = ratemax_numFrames * ifi;


%% ============================  2 Experiment  ============================

try

    % detect and resolve integration issues
    PsychJavaTrouble;
    % supress keyboard input to Matlab command window
    ListenChar(2);
    % hide the mouse
    HideCursor(wPtr);

    %% -------------------------  2.1 Welcome  ----------------------------
    if ~isPrac
        showtext(wPtr, double(welcome_txt), font_size, black_val);
    else
        showtext(wPtr, double(welcome_prc_txt), font_size, black_val);
    end
    Screen('Flip',wPtr);

    KbReleaseWait(); % wait until all keys are released
    while 1
        [keyIsDown, ~, keyCode] = KbCheck;
        if (keyIsDown && size(find(keyCode), 2) == 1)
            if keyCode(space_key)
                break;
            elseif keyCode(escape_key)
                sca; ListenChar(0); error('experiment aborted by user');
            end
            KbReleaseWait();
        end
    end

    %% -------------------  2.2 Block and trial loop  ---------------------

    % data to write
    presskey = []; 
    choice = [];
    rating = [];
    feedback = [];
    chotiming = []; % with format of [sti_StartTime, cho_Time, cho_RT]
    ratetiming = []; % with format of [rate_StartTime, rate_Time, rate_RT]
    rec_DU = nan;
    rectiming = []; % with format of [mic_StartTime, prs_StartTime, rec_duration]
    accuracy = []; % accuracy per category of each block, with format of [a1,a2,b1,b2]
    dprime = []; % dprime per category of each block, with format of [a1,a2,b1,b2]

    % task finish index
    finish = zeros(maxBlock+1,1);

    % load textures
    Screen('BlendFunction', wPtr, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');
    green_texture = Screen('MakeTexture', wPtr, img_feat_cell{1});
    yellow_texture = Screen('MakeTexture', wPtr, img_feat_cell{2});
    pink_texture = Screen('MakeTexture', wPtr, img_feat_cell{3});
    blue_texture = Screen('MakeTexture', wPtr, img_feat_cell{4});
    center_texture = Screen('MakeTexture', wPtr, img_center);


    % ###########################  block loop  ############################
    
    for iBlock = 1:maxBlock
        if finish(iBlock) == 1
            break;
        else
            % ######################  trial loop  #########################
            for iTrial = trl_start(end):trl_end(end)
                stiID = sti_order(iTrial);
                this_info = sti_info(stiID,:);

                varNames = {'green_length','yellow_length','pink_length','blue_length'};
                this_info_norm = array2table(zeros(1, length(varNames)), 'VariableNames', varNames);
         
                for i = 1:4
                    feat_name = string(this_info.(sprintf('feature%d_name', i))); 
                    feat_val  = this_info.(sprintf('feature%d', i));
                    
                    target_col = feat_name + "_length";
                    this_info_norm.(target_col) = feat_val;
                end

                this_info_real = transform_to_real(this_info_norm, feature_range, angle);
                [feature_rects, dots_xys] = screen_xys(this_info_real,center_x,center_y, img_width_ratio, transform_to_pix_rate);
                               
                % ----- 1) show fixation
                Screen('FillOval', wPtr, fix_color, fix_position);
                Screen('DrawingFinished', wPtr); 
                [~, fix_StartTime] = Screen('Flip', wPtr);

                fix_duration = fix_fixed + fix_jitter*rand;
                fix_EndTime = fix_StartTime + fix_duration;
                while GetSecs < fix_EndTime
                end

                % ----- 2) show stimuli figure and options
                Screen('DrawTexture', wPtr, center_texture, [], center_rect);
                Screen('DrawTexture', wPtr, green_texture, [], feature_rects(1,:), rotation_angles(1));
                Screen('DrawTexture', wPtr, yellow_texture, [], feature_rects(2,:), rotation_angles(2));
                Screen('DrawTexture', wPtr, pink_texture, [], feature_rects(3,:), rotation_angles(3));
                Screen('DrawTexture', wPtr, blue_texture, [], feature_rects(4,:), rotation_angles(4));
                Screen('DrawDots', wPtr, dots_xys, dot_size, dot_color, [0,0], 2);

                showoptions(wPtr, opt_xys, opt_width, opt_color, opt_txt, opt_txt_xys, opt_txt_size);
                Screen('DrawingFinished', wPtr);
        
                [~, sti_StartTime] = Screen('Flip', wPtr);

                % ----- 3) detect response
                KbReleaseWait();
                while 1
                    [keyIsDown, ~, keyCode] = KbCheck;
                    if (keyIsDown && size(find(keyCode), 2) == 1)
                        if condition == 1 && (keyCode(Key2) || keyCode(Key3))
                            cho_Time = GetSecs;
                            break;
                        elseif condition > 1 && (keyCode(Key1) || keyCode(Key2) || keyCode(Key3) || keyCode(Key4))
                            cho_Time = GetSecs;
                            break;
                        elseif keyCode(escape_key)
                            save([dataname, '.mat'], 'behavior');
                            sca; ListenChar(0); error('experiment aborted by user');
                        end
                    end
                end
                KbReleaseWait();

                % ----- 4) record response
                % RT
                cho_RT = cho_Time - sti_StartTime;
                chotiming = cat(1, chotiming, [sti_StartTime, cho_Time, cho_RT]);

                % choice
                if condition == 1
                    if (ismember({KbName(keyCode)}, Keys(2)))
                        presskey = cat(1, presskey, 1);
                        choice = cat(1, choice, keys_category(1));
                    elseif (ismember({KbName(keyCode)}, Keys(3)))
                        presskey = cat(1, presskey, 2);
                        choice = cat(1, choice, keys_category(2));
                    end
                elseif condition > 1
                    if (ismember({KbName(keyCode)}, Keys(1)))
                        presskey = cat(1, presskey, 1);
                    choice = cat(1, choice, keys_category(1));
                    elseif (ismember({KbName(keyCode)}, Keys(2)))
                        presskey = cat(1, presskey, 2);
                        choice = cat(1, choice, keys_category(2));
                    elseif (ismember({KbName(keyCode)}, Keys(3)))
                        presskey = cat(1, presskey, 3);
                        choice = cat(1, choice, keys_category(3));
                    elseif (ismember({KbName(keyCode)}, Keys(4)))
                        presskey = cat(1, presskey, 4);
                        choice = cat(1, choice, keys_category(4));
                    end
                end

                % ----- 5) show selected option
                % stimuli
                Screen('DrawTexture', wPtr, center_texture, [], center_rect);
                Screen('DrawTexture', wPtr, green_texture, [], feature_rects(1,:), rotation_angles(1));
                Screen('DrawTexture', wPtr, yellow_texture, [], feature_rects(2,:), rotation_angles(2));
                Screen('DrawTexture', wPtr, pink_texture, [], feature_rects(3,:), rotation_angles(3));
                Screen('DrawTexture', wPtr, blue_texture, [], feature_rects(4,:), rotation_angles(4));
                Screen('DrawDots', wPtr, dots_xys, dot_size, dot_color, [0,0], 2);

                % higlighed by choice
                showselectedoption(wPtr, condition, opt_xys, sopt_color, KbName(keyCode), Keys);
                % in their options
                showoptions(wPtr, opt_xys, opt_width, opt_color, opt_txt, opt_txt_xys, opt_txt_size);
                Screen('DrawingFinished', wPtr);

                [~, sopt_StartTime] = Screen('Flip', wPtr);
                sopt_EndTime = sopt_StartTime + sopt_duration;

                while GetSecs < sopt_EndTime
                end

                % Short blank
                Screen('FillRect', wPtr, bg_color);
                Screen('Flip', wPtr);
                WaitSecs(blank_numSecs);

                % ----- 6) Confidence rating (max 30s)
                Screen('TextSize', wPtr, font_size);
                for i = scale
                    DrawFormattedText(wPtr, double(num2str(i)), w_rect(3)/12*(i+2)+w_rect(3)/24, w_rect(4)/2, 0);
                    if i == 1
                        DrawFormattedText(wPtr, double('非常不确定'), w_rect(3)/12*(i+1)+w_rect(3)/24, w_rect(4)/2+w_rect(4)/12, 0);
                    elseif i == length(scale)
                        DrawFormattedText(wPtr, double('非常确定'), w_rect(3)/12*(i+2)+w_rect(3)/24, w_rect(4)/2+w_rect(4)/12, 0);
                    end
                end
                [~, rate_StartTime] = Screen('Flip', wPtr);

                KbReleaseWait();
                rate_EndTime = rate_StartTime + ratemax_duration;
                thisrate = nan;
                rate_Time = nan;
                rate_RT = nan;
                while GetSecs <= rate_EndTime
                    [keyIsDown, ~, keyCode] = KbCheck;
                    if keyIsDown
                        pressedKeys = find(keyCode);
                        if numel(pressedKeys) == 1
                            pk = pressedKeys(1);
                            if any(ismember(pk, rating_keys))
                                rate_Time = GetSecs;
                                rate_RT = rate_Time - rate_StartTime;

                                idx = find(rating_keys == pk, 1);
                                if ~isempty(idx)
                                    thisrate = idx;
                                end
                                break;
                            elseif keyCode(escape_key)
                                save([dataname, '.mat'], 'behavior');
                                sca; ListenChar(0); error('experiment aborted by user');
                            end
                        end
                    end
                end
                KbReleaseWait();
                rating = cat(1, rating, thisrate);
                ratetiming = cat(1, ratetiming, [rate_StartTime, rate_Time, rate_RT]);

                % Short blank
                Screen('FillRect', wPtr, bg_color);
                Screen('Flip', wPtr);
                WaitSecs(blank_numSecs);

                % ----- 7) sound recording preparation
                if IsRecording
                    showtext(wPtr, double(rec_txt), font_size, 0);
                    Screen('Flip', wPtr);
    
                    KbReleaseWait();
                    while 1
                        [keyIsDown, ~, keyCode] = KbCheck;
                        if (keyIsDown && size(find(keyCode), 2) == 1)
                            if keyCode(space_key)
                                break;
                            elseif keyCode(escape_key)
                                save([dataname, '.mat'], 'behavior');
                                sca; ListenChar(0); error('experiment aborted by user');
                            end
                            KbReleaseWait();
                        end
                    end
    
                    % ----- 8) sound recording 
                    % show microphone symbol
                    Screen('DrawTexture', wPtr, mic_texture, [], mic_rect, 0);
                    Screen('DrawingFinished', wPtr);
                    [~, mic_StartTime] = Screen('Flip', wPtr);
    
                    % start recording
                    record(recorder);
    
                    % stop recording with button press
                    prs_StartTime = 0;
                    while 1
                        [keyIsDown, secs, keyCode] = KbCheck;
                        if (keyIsDown && size(find(keyCode), 2) == 1)                  
                            if keyCode(space_key)
                                if prs_StartTime == 0
                                    prs_StartTime = secs; % record the time when the spacebar was first pressed
                                elseif secs - prs_StartTime >= prs_duration
                                    break; % break when the spacebar is pressed continuously for certain duration
                                end
                            elseif keyCode(escape_key)
                                save([dataname, '.mat'], 'behavior');
                                sca; ListenChar(0); error('experiment aborted by user');
                            end
                        else
                            prs_StartTime = 0; % reset the start time if the spacebar is released
                        end
                    end
                    stop(recorder);
    
                    % save audio data as .wav file
                    audioData = getaudiodata(recorder, 'double');
                    audiowrite([fullfile(recfolder, num2str(iTrial)), '.wav'], audioData, 44100);
    
                    % get the recording duration
                    rec_DU = prs_StartTime - mic_StartTime;
                    rectiming = cat(1, rectiming, [mic_StartTime, prs_StartTime, rec_DU]);
    
                    % add blank interval
                    Screen('FillRect', wPtr, bg_color);
                    Screen('Flip',wPtr);
                    WaitSecs(blank_numSecs);
                end

                % ----- 9) show feedback
                probCat1 = sti_info.prob_category1(stiID);
                probCat2 = sti_info.prob_category2(stiID);
                [thisfdb, thisfdb_prob, thischosen_prob] = probabilistic_feedback(choice(end), probCat1, probCat2);
                feedback = cat(1, feedback, thisfdb);

                showtext(wPtr, double([num2str(thisfdb)]), 70, 0);
                Screen('Flip', wPtr);

                KbReleaseWait();
                while 1
                    [keyIsDown, ~, keyCode] = KbCheck;
                    if (keyIsDown && size(find(keyCode), 2) == 1)
                        if keyCode(space_key)
                            break;
                        elseif keyCode(escape_key)
                            save([dataname, '.mat'], 'behavior');
                            sca; ListenChar(0); error('experiment aborted by user');
                        end
                    end
                end
                KbReleaseWait();

                % ----- 10) data recording
                behavior.choice = choice;
                behavior.presskey = presskey;
                behavior.rating = rating;
                behavior.feedback = feedback;
                behavior.chotiming = chotiming;
                behavior.ratetiming = ratetiming;
                if IsRecording
                    behavior.rectiming = rectiming;
                end
                
                fprintf(fid, '%i,%i,%i,%i,%i,%3.3f,%3.3f,%i,%i,%i,%i,%3.3f,%3.3f,%3.3f\n', ...
                    condition, sessionID, iBlock, iTrial, stiID, probCat1, probCat2, ...
                    choice(end), presskey(end), rating(end), feedback(end), cho_RT, rate_RT, rec_DU);

                % ----- 11) rest after several trials
                if ~isPrac
                    if any(iToR(:) == mod(iTrial, nTpB))
                        ts = Screen('Flip', wPtr);
                        for iTime = 1:rest_numSecs
                            showtext(wPtr, double(sprintf(rest_txt, rest_numSecs+1-iTime)), font_size, 0);
                            ts = Screen('Flip', wPtr, ts+1);
                        end
                        WaitSecs(1);
                        Screen('Flip', wPtr, ts+1);

                        save([dataname, '.mat'], 'behavior');

                        showtext(wPtr, double(restend_txt), font_size, 0);
                        Screen('Flip', wPtr);
    
                        KbReleaseWait();
                        while 1
                            [keyIsDown, ~, keyCode] = KbCheck;
                            if (keyIsDown && size(find(keyCode), 2) == 1)
                                if keyCode(space_key)
                                    break;
                                elseif keyCode(escape_key)
                                    save([dataname, '.mat'], 'behavior');
                                    sca; ListenChar(0); error('experiment aborted by user');
                                end
                                KbReleaseWait();
                            end
                        end
                    end
                end
            end
            % ####################  trial loop end  #######################

            % #################  judge, generate, and rest  ###############
            if ~isPrac
                % calculate accuracy
                thisblock_fdb = feedback(trl_start(end):trl_end(end));
                thisblock_amb = sti_ambiguous(trl_start(end):trl_end(end));
                thisblock_acc = sum(thisblock_fdb(thisblock_amb==0))/sum(thisblock_amb==0);

                accuracy = cat(1, accuracy, thisblock_acc);
                behavior.accuracy = accuracy;

                save([dataname, '.mat'], 'behavior');


                showtext(wPtr, double(sprintf(blockend_txt, thisblock_acc)), font_size, 0);
                Screen('Flip', wPtr);

                KbReleaseWait();
                while 1
                    [keyIsDown, ~, keyCode] = KbCheck;
                    if (keyIsDown && size(find(keyCode), 2) == 1)
                        if keyCode(space_key)
                            break;
                        elseif keyCode(escape_key)
                            save([dataname, '.mat'], 'behavior');
                            sca; ListenChar(0); error('experiment aborted by user');
                        end
                        KbReleaseWait();
                    end
                end

                % judge whether finish
                if thisblock_acc > accstop
                    finish(iBlock+1) = 1;
                else
                    finish(iBlock+1) = 0;
                end

                if finish(iBlock+1) == 1 || iBlock == maxBlock
                    break;
                else
                    % sampling for the next block
                    trl_start = cat(1, trl_start, nTpB*(iBlock)+1);
                    trl_end = cat(1, trl_end, nTpB*(iBlock+1));
                    sti_info = cat(1, sti_info, sample_2_beta(feature_names, nTpB, prototype1, prototype2, beta));
                    writetable(sti_info, fullfile(folders{1}, [taskID,'_',num2str(subID),'_',num2str(sessionID), '_sti.csv']));

                    % rest before next block
                    ts = Screen('Flip', wPtr);
                    for iTime = 1:rest_numSecs
                        % show countdown
                        showtext(wPtr, double(sprintf(rest_txt, rest_numSecs+1-iTime)), font_size, 0);
                        ts = Screen('Flip', wPtr, ts+1);
                    end
                    WaitSecs(1);
                    Screen('Flip', wPtr, ts+1);

                    
                    % get new stimulus matrix and order
                    sti_order = cat(1, sti_order, Shuffle(trl_start(end):trl_end(end))'); 
                    sti_probcat1 = sti_info.prob_category1(sti_order);             
                    sti_probcat2 = sti_info.prob_category2(sti_order);
                    sti_ambiguous = sti_info.ambiguous(sti_order);

                    behavior.sti_info = sti_info;
                    behavior.trl_start = trl_start;
                    behavior.trl_end = trl_end;
                    behavior.sti_order = sti_order;
                    behavior.sti_probcat1 = sti_probcat1;
                    behavior.sti_probcat2 = sti_probcat2;
                    behavior.sti_ambiguous = sti_ambiguous;
                    save([dataname, '.mat'], 'behavior');

                    % ask if ends rest
                    showtext(wPtr, double(restend_txt), font_size, 0);
                    Screen('Flip', wPtr);

                    KbReleaseWait();
                    while 1
                        [keyIsDown, ~, keyCode] = KbCheck;
                        if (keyIsDown && size(find(keyCode), 2) == 1)
                            if keyCode(space_key)
                                break;
                            elseif keyCode(escape_key)
                                save([dataname, '.mat'], 'behavior');
                                sca; ListenChar(0); error('experiment aborted by user');
                            end
                            KbReleaseWait();
                        end
                    end
                end
            end
        end
    end
    % #########################  block loop end  ##########################
    save([dataname, '.mat'], 'behavior');

    WaitSecs(1);

    %% --------------------------  2.3 Ending  ----------------------------
    if ~isPrac
        avgacc = nanmean(accuracy,'all'); % convert accuracy to earning
        behavior.avgacc = avgacc;
        save([dataname, '.mat'], 'behavior');

        showtext(wPtr, double(sprintf(end_txt,avgacc)), font_size, 0);
    else
        showtext(wPtr, double(end_prc_txt), font_size, 0);
    end

    Screen('Flip', wPtr);

    KbReleaseWait();
    while 1
        [ keyIsDown, seconds, keyCode] = KbCheck;
        if (keyIsDown && size(find(keyCode), 2) == 1)
            if keyCode(escape_key)
                break;
            end
            KbReleaseWait();
        end
    end

    Screen('CloseAll');
    ShowCursor;
    ListenChar(0); % restore keyboard response in Matlab command window

catch
    Screen('CloseAll');
    ShowCursor;
    ListenChar(0); % restore keyboard response in Matlab command window
    psychrethrow(psychlasterror);
end


%% =========================  3 Utility function  =========================
%% Get probabilistic feedback for binary prototype-based task
function [thisfdb, pCorrect, pChosen] = probabilistic_feedback(choice, probCat1, probCat2)
if choice == 1
    pChosen = probCat1;
elseif choice == 2
    pChosen = probCat2;
else
    pChosen = nan;
end

pCorrect = pChosen;
if isnan(pCorrect)
    thisfdb = nan;
else
    thisfdb = rand < pCorrect;
end
end

%% Get the number of each category for the next block
function number = generate(condition, nTpB, dprime, beta)
inv = 6 - dprime;
percent = exp(beta.*inv)/sum(exp(beta.*inv));

for n = 1:size(percent,2)
    number(n) = floor(percent(n)/sum(percent) * nTpB);
end

if sum(number == min(number)) == 3
    number(number == min(number)) = number(number == min(number)) + floor((nTpB - sum(number))/3);
    if mod((nTpB - sum(number)), 3) == 1
        number(find(number == min(number), 1)) = number(find(number == min(number), 1)) + 1;
    elseif mod((nTpB - sum(number)), 3) == 2
        indices = find(number == min(number), 3);
        number(indices(1)) = number(indices(1)) + 1;
        number(indices(2)) = number(indices(2)) + 1;
    end
elseif sum(number == min(number)) == 2
    indices = find(number == min(number), 2);
    gap = nTpB - sum(number);
    number(indices(1)) = number(indices(1)) + floor(gap/2);
    number(indices(2)) = number(indices(2)) + ceil(gap/2);
elseif sum(number == min(number)) == 1
    number(number == min(number)) = number(number == min(number)) + nTpB - sum(number);
end

if condition == 1
    number = repelem(number/2, 1, 2);
end
end


%% Display text
function showtext(w, text, size, color)
Screen('TextSize', w, size);
DrawFormattedText(w, text, 'center', 'center', color);
end

%% Display options
function showoptions(w, opt_xys, opt_width, opt_color, opt_txt, opt_txt_xys, opt_txt_size)
Screen('FrameOval', w, opt_color, opt_xys, opt_width);

Screen('TextSize', w, opt_txt_size);
textBounds = Screen('TextBounds', w, char(opt_txt(1)));
txt_x = opt_txt_xys(1) - textBounds(3) / 2;
txt_y = opt_txt_xys(2) - textBounds(4) / 2;
interval = opt_xys(1,2) - opt_xys(1,1);
for keys = opt_txt
    Screen('DrawText', w, char(keys), txt_x, txt_y);
    txt_x = txt_x + interval;
end
end

%% Display selected option
function showselectedoption(w, condition, opt_xys, sopt_color, keyPress, Keys)
if condition == 1
    if ismember({keyPress}, Keys(2))  % press the F key
        Screen('FillOval', w, sopt_color, opt_xys(:,1));
    elseif ismember({keyPress}, Keys(3)) % press the J key
        Screen('FillOval', w, sopt_color, opt_xys(:,2));
    end
else
    if ismember({keyPress}, Keys(1))  % press the F key
        Screen('FillOval', w, sopt_color, opt_xys(:,1));
    elseif ismember({keyPress}, Keys(2)) % press the G key
        Screen('FillOval', w, sopt_color, opt_xys(:,2));
    elseif ismember({keyPress}, Keys(3)) % press the H key
        Screen('FillOval', w, sopt_color, opt_xys(:,3));
    elseif ismember({keyPress}, Keys(4)) % press the J key
        Screen('FillOval', w, sopt_color, opt_xys(:,4));
    end
end
end