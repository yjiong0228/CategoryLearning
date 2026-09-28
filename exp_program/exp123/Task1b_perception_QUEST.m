% Perception Calibration - quest
% 2025.9.16

%% ==========================  1.1 Initializing  ==========================
clear all;
  
taskID = 'Task1b';

% input information
subID = input('Input the subject ID -> ', 's');
subID = str2double(subID);
behavior.subID = subID;

% Create folder
data_folder = fullfile(taskID, 'Data');
if ~exist(data_folder,'dir')
    mkdir(data_folder);
end
dataname = fullfile(data_folder, [taskID, '_', num2str(subID), '_bhv']);

% open .dat file for writing
fid = fopen([dataname, '.csv'], 'wt');
fprintf(fid, '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n', ...
    'subID','condition','level','iTrial','orien','standard_real','compare_real','diff_real', ...
    'standard_norm','compare_norm','diff_norm', ...
    'truth','choice','feedback','cho_RT','threshold_mean','threshold_sd'); % headers

%% =======================  1.2 Task organization  ========================

% set the unique random seed based on subID
s = RandStream('mt19937ar','Seed',subID*2025);
RandStream.setGlobalStream(s);

nCon = 4;
nLevel = 5;

nBlock = nCon * nLevel;

con_seq = Shuffle(repmat(1:nCon, 1));
level_seq = [Shuffle(repmat(1:nLevel, 1)), Shuffle(repmat(1:nLevel, 1)),...
    Shuffle(repmat(1:nLevel, 1)), Shuffle(repmat(1:nLevel, 1))]';

blockInfo = table( ...
    (1:nBlock)', ...
    repelem(con_seq,nLevel)', ...
    level_seq, ...
    'VariableNames', {'Block', 'con_seq', 'level_seq'} ...
);


maxTrials = 30; % trial number per each level


% stimulus feature range
feature_range = struct();
feature_range.body_length = 1;
feature_range.neck_length = [feature_range.body_length/4, feature_range.body_length*5/4];
feature_range.head_length = [feature_range.body_length/4, feature_range.body_length*5/4];
feature_range.leg_length = [feature_range.body_length/4, feature_range.body_length*5/4];
feature_range.tail_length = [feature_range.body_length/4, feature_range.body_length*5/4];
feature_range.neck_angle = [-pi/3, 0];
feature_range.head_angle = [0, pi/3];
feature_range.leg_angle = [pi*11/36, pi*17/36];
feature_range.tail_angle = [-pi/3, 0];

% initialize info_norm
info_norm = table( ...
    0.5, 0.5, 0.5, 0.5, ...  
    0.5, 0.5, 0.5, 0.5, ... 
    'VariableNames', { ...
        'neck_length', 'head_length', 'leg_length', 'tail_length', ...
        'neck_angle', 'head_angle', 'leg_angle', 'tail_angle' ...
    } ...
);

behavior.blockInfo = blockInfo;
behavior.feature_range = feature_range;


%% ======================  1.3 Response key settings  =====================

KbName('UnifyKeyNames');

left_key = KbName('f');
right_key = KbName('j');

space_key = KbName('SPACE');
escape_key = KbName('ESCAPE');

RestrictKeysForKbCheck([left_key, right_key, space_key, escape_key]);


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
font_size = 2 * deg2pix;
fix_radius = 0.003 * w_width;
fix_position = [center_x-fix_radius; center_y-fix_radius;...
    center_x+fix_radius; center_y+fix_radius];

% transform stimulus
body_visual_angle = 3.8; % visual angle of stimulus width
body_length_pix = round(body_visual_angle * deg2pix);

transform_to_pix_rate = body_length_pix / feature_range.body_length;

img_width_rate = 4; % stimulus image width / body length
sti_width = body_length_pix * img_width_rate;
ratio = 0.75; % width-height ratio of stimuli figure
sti_height = round(sti_width * ratio);
sti_rect = [center_x-sti_width/2, center_y-sti_height/2, center_x+sti_width/2, center_y+sti_height/2];

% draw stimulus settings
dot_size = 8;
line_width = 10;

% transform mask
nlines = 60;
xmin = round(sti_rect(1));
xmax = round(sti_rect(3));
ymin = round(sti_rect(2));
ymax = round(sti_rect(4));

% basic colors
black_val = BlackIndex(screenID);
white_val = WhiteIndex(screenID);
grey_val = white_val / 2;
bg_color = grey_val;
fix_color = [0 115 0];


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


%% =========================  1.6 Introductions  ==========================

welcome_txt = '任务1b\n\n准备好后请按空格键开始';
condition_names = {'脖子','头','腿','尾巴'};
level_begin_txt = '现在%s测试你对于【%s】的长度感知\n\n准备好后请按空格键开始';
level_rest_txt = '休息一下，倒计时%i秒';
end_txt = '任务1b到此结束\n\n你的正确率为%3.3f';


%% ===================  1.7 Experiment timing settings  ===================

fix_numSecs = 0.6; % fixation duration
fixj_numSecs = 0.25; % fixation jitter
sti_numSecs = 0.8; % stimulus duration
mask_numSecs = 0.15; % mask duration
iti_numSecs = 0.5; % iti duration
rest_numSecs = 30; % rest between blocks


ifi = Screen('GetFlipInterval', wPtr); % the vertical refresh rate of the monitor

fix_numFrames = round(fix_numSecs / ifi);
fix_fixed = fix_numFrames * ifi;

fixj_numFrames = round(fixj_numSecs / ifi);
fix_jitter = fixj_numFrames * ifi;

sti_numFrames = round(sti_numSecs / ifi);
sti_duration = sti_numFrames * ifi;

mask_numFrames = round(mask_numSecs / ifi);
mask_duration = mask_numFrames * ifi;

iti_numFrames = round(iti_numSecs / ifi);
iti_duration = iti_numFrames * ifi;


%% ============================  2 Experiment  ============================

try

    % detect and resolve integration issues
    PsychJavaTrouble;
    % supress keyboard input to Matlab command window
    ListenChar(2);
    % hide the mouse
    HideCursor(wPtr);

    %% -------------------------  2.1 Welcome  ----------------------------

    % press practice
%     if isPrac
%         practice(wPtr, w_rect, option, font_size, choice_keys, escape_key)
%     end
    
    % welcome screen
    Screen('TextSize', wPtr, font_size);
    DrawFormattedText(wPtr, double(welcome_txt), 'center', 'center', 0);
    Screen('DrawingFinished', wPtr);
    Screen('Flip', wPtr);

    KbReleaseWait();
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
    orien = [];
    standard_real = [];
    compare_real = [];
    diff_real = [];
    standard_norm = [];
    compare_norm = [];
    diff_norm = [];
    threshold_mean = [];
    threshold_sd = [];

    truth = []; % 1 = sti1 > sti2; 2 = sti1 < sti2
    choice = []; % 1 = sti1 > sti2; 2 = sti1 < sti2
    chotiming = []; % with format of [sti_StartTime, resp_Time, RT]
    feedback = []; % 1 = right; 0 = wrong
    q = []; % Quest struct


    %% ########################  block loop  ##########################
    for iBlock = 1:nBlock

        iCon = blockInfo.con_seq(iBlock);
        iLevel = blockInfo.level_seq(iBlock);

        fields = fieldnames(feature_range);
        feature_name = fields{iCon+1};
        levels_real = linspace(feature_range.(feature_name)(1), feature_range.(feature_name)(2), nLevel);

        
        this_standard_real = levels_real(iLevel);

 
        % instruction
        if iLevel == level_seq(1)
            txt = double(sprintf(level_begin_txt, '开始', condition_names{iCon}));
        else
            txt = double(sprintf(level_begin_txt, '继续', condition_names{iCon}));
        end

        Screen('TextSize', wPtr, font_size);
        DrawFormattedText(wPtr, txt, 'center', 'center', 0);
        Screen('DrawingFinished', wPtr);
        Screen('Flip', wPtr);

        KbReleaseWait();
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

        % blank screen
        Screen('FillRect', wPtr, bg_color);
        Screen('Flip', wPtr);
        WaitSecs(0.5);



        % initialize QUEST
        thresholdGuess = 0.03; % initial mean
        thresholdGuessSD = 0.03; % initial sd
        pThreshold = 0.7; % accuracy
        beta = 3.5; % slope
        delta = 0.2; % lapse rate
        gamma = 0.5; % guess rate of 2AFC
        
        
        thisq = QuestCreate(thresholdGuess, thresholdGuessSD, pThreshold, beta, delta, gamma);
        thisq.normalizePdf = 1;

        thistruth = NaN;
        thischo = NaN;
        thisfdb = NaN;

        %% #######################  trial loop  #######################
        for iTrial = 1:maxTrials

            % decide orientation randomly
            iOrien = round(rand)*2-1; % 1=right;-1=left
            orien = cat(1,orien,iOrien);
            behavior.orien = orien;

            % get intensity from quest
            this_diff_real = QuestQuantile(thisq);

            % decide direction
            if iLevel == 1
                iDir = 1; % 1=compare > standard
            elseif iLevel == nLevel
                iDir = -1; % -1=compare < standard
            else
                iDir = round(rand)*2-1; % random
            end

            this_compare_real = this_standard_real + iDir * this_diff_real;

            % transform normalized features into real features
            info_norm.body_ori = iOrien;
            info_real = transform_to_real(info_norm,feature_range);

            info_standard_real = info_real;
            info_standard_real.(feature_name) = this_standard_real;

            info_compare_real = info_real;
            info_compare_real.(feature_name) = this_compare_real;


            % transform real features to coordinates in screen
            [linexys_standard, dotxys_standard] = screen_xys(info_standard_real,center_x,center_y,transform_to_pix_rate);
            [linexys_compare, dotxys_compare] = screen_xys(info_compare_real,center_x,center_y,transform_to_pix_rate);


            % decide order of standard and compare stimulus
            isStandardFirst = rand < 0.5;
            if isStandardFirst
                linexys_1 = linexys_standard;
                linexys_2 = linexys_compare;
                dotxys_1 = dotxys_standard;
                dotxys_2 = dotxys_compare;
                if iDir == 1
                    thistruth = 2; 
                else
                    thistruth = 1; 
                end
            else
                linexys_1 = linexys_compare;
                linexys_2 = linexys_standard; 
                dotxys_1 = dotxys_compare;
                dotxys_2 = dotxys_standard; 
                if iDir == 1
                    thistruth = 1; 
                else
                    thistruth = 2; 
                end
            end
            truth = cat(1, truth, thistruth);
            behavior.truth = truth;


            % ----- 1) show cue
            cue_position = fix_position;

            if iCon == 1 % neck
                cue_position(1) = cue_position(1) + iOrien * 0.7 * body_length_pix;
                cue_position(3) = cue_position(3) + iOrien * 0.7 * body_length_pix;
            elseif iCon == 2 % head
                cue_position(1) = cue_position(1) + iOrien * 1.3 * body_length_pix;
                cue_position(3) = cue_position(3) + iOrien * 1.3 * body_length_pix;

            elseif iCon == 4 % tail
                cue_position(1) = cue_position(1) - iOrien * 0.7 * body_length_pix;
                cue_position(3) = cue_position(3) - iOrien * 0.7 * body_length_pix;
            end

            Screen('FillOval', wPtr, fix_color, cue_position);
            Screen('DrawingFinished', wPtr);
            [~, fix_StartTime] = Screen('Flip', wPtr);
    
            fix_duration = fix_fixed + fix_jitter*rand;
            fix_EndTime = fix_StartTime + fix_duration;
            while GetSecs < fix_EndTime
            end
    

            % ----- 2) show stimulus 1
            Screen('DrawLines', wPtr, linexys_1, line_width, black_val);
            Screen('DrawDots', wPtr, dotxys_1, dot_size, black_val, [0,0], 2);
            Screen('DrawingFinished', wPtr);
            [~, sti_1_StartTime] = Screen('Flip', wPtr);
    
            sti_1_EndTime = sti_1_StartTime + sti_duration;
            while GetSecs < sti_1_EndTime
            end
    

            % ----- 3) show mask 1
            Screen('FillRect', wPtr, bg_color); 
            for i = 1:nlines
                x1 = randi([xmin, xmax]);
                y1 = randi([ymin, ymax]);
                x2 = randi([xmin, xmax]);
                y2 = randi([ymin, ymax]);
                Screen('DrawLine', wPtr, 0, x1, y1, x2, y2, line_width);
            end

            [~, mask_StartTime] = Screen('Flip', wPtr);
    
            mask_EndTime = mask_StartTime + mask_duration;
            while GetSecs < mask_EndTime
            end
    
    
            % ----- 4) show stimulus 2
            Screen('DrawLines', wPtr, linexys_2, line_width, black_val);
            Screen('DrawDots', wPtr, dotxys_2, dot_size, black_val, [0,0], 2);
            Screen('DrawingFinished', wPtr);
            [~, sti_2_StartTime] = Screen('Flip', wPtr);
    
            sti_2_EndTime = sti_2_StartTime + sti_duration;
            while GetSecs < sti_2_EndTime
            end
    

            % ----- 5) show mask 2
            Screen('FillRect', wPtr, bg_color); 
            for i = 1:nlines
                x1 = randi([xmin, xmax]);
                y1 = randi([ymin, ymax]);
                x2 = randi([xmin, xmax]);
                y2 = randi([ymin, ymax]);
                Screen('DrawLine', wPtr, 0, x1, y1, x2, y2, line_width);
            end
            Screen('DrawingFinished', wPtr);
            [~, mask_StartTime] = Screen('Flip', wPtr);
    
            mask_EndTime = mask_StartTime + mask_duration;
            while GetSecs < mask_EndTime
            end
    

            % ----- 6) show options and wait for response
            Screen('TextSize', wPtr, font_size);
            DrawFormattedText(wPtr, double('    F\n\n第1只更长'), w_rect(3)/5, w_rect(4)/2, 0);
            DrawFormattedText(wPtr, double('    J\n\n第2只更长'), w_rect(3)*3/5, w_rect(4)/2, 0);
            Screen('DrawingFinished', wPtr);

            [~, option_StartTime] = Screen('Flip', wPtr);

            KbReleaseWait();
            while 1
                [keyIsDown, ~, keyCode] = KbCheck;
                if (keyIsDown && size(find(keyCode), 2) == 1)
                    if keyCode(left_key) || keyCode(right_key)

                        cho_Time = GetSecs;
                        cho_RT = cho_Time - option_StartTime;
        
                        if keyCode(left_key)
                            thischo = 1; % stimulus 1 longer
                        elseif keyCode(right_key)
                            thischo = 2; % stimulus 2 longer
                        end

                        break;
                    elseif keyCode(escape_key)
                        save([dataname, '.mat'], 'behavior');
                        sca; ListenChar(0); error('experiment aborted by user');
                    end
                end
            end
            KbReleaseWait();

    
            choice = cat(1, choice, thischo);
            chotiming = cat(1, chotiming, [option_StartTime, cho_Time, cho_RT]);
    
            if thischo == thistruth
                thisfdb = 1;
            else
                thisfdb = 0;
            end
            feedback = cat(1, feedback, thisfdb);
    
    
            % ----- 7) ITI - blank screen
            Screen('FillRect', wPtr, bg_color);
            [~, iti_StartTime] = Screen('Flip',wPtr);
    
            KbReleaseWait();
            iti_EndTime = iti_StartTime + iti_duration;
            while GetSecs < iti_EndTime
                [keyIsDown, ~, keyCode] = KbCheck;
                if (keyIsDown && size(find(keyCode), 2) == 1)
                    if keyCode(escape_key)
                        sca; ListenChar(0); error('experiment aborted by user');
                    end
                end
            end
            KbReleaseWait();


            % ----- 8) update
            thisq = QuestUpdate(thisq, this_diff_real, thisfdb);
            q = cat(1, q, thisq);
            behavior.q = q;

            this_threshold_mean_real = QuestMean(thisq);
            this_threshold_sd_real = QuestSd(thisq);


            % ----- 9) data recording
            behavior.choice = choice;
            behavior.feedback = feedback;
            behavior.chotiming = chotiming;
    

            % transform real features to normalized features
            this_standard_norm = transform_to_norm(this_standard_real, feature_range.(feature_name));
            this_compare_norm = transform_to_norm(this_compare_real, feature_range.(feature_name));
            this_diff_norm = this_compare_norm - this_standard_norm;
            
            this_threshold_mean = this_threshold_mean_real/(feature_range.(feature_name)(2)-feature_range.(feature_name)(1));
            this_threshold_sd = this_threshold_sd_real/(feature_range.(feature_name)(2)-feature_range.(feature_name)(1));


            fprintf(fid, '%i,%i,%i,%i,%i,%3.3f,%3.3f,%3.3f,%3.3f,%3.3f,%3.3f,%i,%i,%i,%3.3f,%3.3f,%3.3f\n', ...
            subID, iCon, iLevel, iTrial, iOrien, this_standard_real, this_compare_real, iDir*this_diff_real, ...
            this_standard_norm, this_compare_norm, this_diff_norm, thistruth, thischo, thisfdb, cho_RT, this_threshold_mean, this_threshold_sd);

            
            standard_real = cat(1,standard_real,this_standard_real);
            compare_real = cat(1,compare_real,this_compare_real);
            diff_real = cat(1,diff_real,iDir*this_diff_real);
            standard_norm = cat(1,standard_norm,this_standard_norm);
            compare_norm = cat(1,compare_norm,this_compare_norm);
            diff_norm = cat(1,diff_norm,iDir*this_diff_norm);
            threshold_mean = cat(1,threshold_mean,this_threshold_mean);
            threshold_sd = cat(1,threshold_sd,this_threshold_sd);

            behavior.standard_real = standard_real;
            behavior.compare_real = compare_real;
            behavior.diff_real = diff_real;
            behavior.standard_norm = standard_norm;
            behavior.compare_norm = compare_norm;
            behavior.diff_norm = diff_norm;
            behavior.threshold_mean = threshold_mean;
            behavior.threshold_sd = threshold_sd;

        end
        % ######################  trial loop end  #####################

        % ----- 10) rest
        if iBlock ~= nBlock
            ts = Screen('Flip', wPtr);
            for iTime = 1:rest_numSecs
                txt = double(sprintf(level_rest_txt, rest_numSecs+1-iTime));
                Screen('TextSize', wPtr, font_size);
                DrawFormattedText(wPtr, txt, 'center', 'center', 0);
                Screen('DrawingFinished', wPtr);
                ts = Screen('Flip', wPtr, ts+1);
            end
            WaitSecs(1);
            Screen('Flip', wPtr, ts+1);
        end

        save([dataname, '.mat'], 'behavior');
    end
    % ########################  condition loop end  #######################


    WaitSecs(1);

    %% --------------------------  2.3 Ending  ----------------------------
    accuracy = sum(feedback)/length(feedback);

    Screen('TextSize', wPtr, font_size);
    DrawFormattedText(wPtr, double(sprintf(end_txt, accuracy)), 'center', 'center', 0);
    Screen('DrawingFinished', wPtr);
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



%% ------------------------- related functions ----------------------------
function x_norm = transform_to_norm(x, range)
    x_norm = (x - range(1)) / (range(2) - range(1));
end

