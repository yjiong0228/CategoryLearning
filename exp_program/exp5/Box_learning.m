%% ==========================  1.1 Initializing  ==========================
% - Stimulus rendering rule: +1 = white square, -1 = black square.

clear all;

% initialize recording device
IsRecording = 1;
if IsRecording
    fs = 44100;
    nbits = 16;
    nchannels = 1;
    recorder = audiorecorder(fs, nbits, nchannels);
end

ruleID = 1;

% input information
subID = input('Input the subject ID -> ', 's');
isPrac = input('Input the experiment mode -> ', 's');

subID = str2double(subID);
isPrac = str2double(isPrac);
behavior.subID = subID;
behavior.isPrac = isPrac;

% Create folders
folders = {fullfile(['Rule',num2str(ruleID)],'Data'), fullfile(['Rule',num2str(ruleID)],'Recording')};
for i = 1:numel(folders)
    if ~exist(folders{i}, 'dir')
        mkdir(folders{i});
    end
end

if isPrac
    sessionID = 0;
    dataname = fullfile(folders{1}, [num2str(subID), '_prac_bhv']);
else
    sessionID = input('Input the session ID -> ', 's');
    sessionID = str2double(sessionID);
    behavior.sessionID = sessionID;
    dataname = fullfile(folders{1}, [num2str(subID), '_', num2str(sessionID), '_bhv']);
end

% Create recording folders
if ~isPrac
    recfolder = fullfile(folders{2}, [num2str(subID),'_',num2str(sessionID)]);
else
    recfolder = fullfile(folders{2}, num2str(999));
end
if ~exist(recfolder, 'dir')
    mkdir(recfolder);
end

fid = fopen([dataname, '.csv'],'wt');
fprintf(fid, '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n', ...
    'ruleID','iSession','iBlock','iTrial','stiID','category', ...
    'choice','presskey','rating','feedback','choRT','rateRT','recDU');

%% =======================  1.2 Task organization  ========================

% set the unique random seed based on subID
s = RandStream('mt19937ar','Seed',subID*202604);
RandStream.setGlobalStream(s);

% task parameters
maxBlock = [1, 5]; % maximum block number, separately for practice and formal task
nTpB = [8, 64]; % trial number per block, separately for practice and formal task
iToR = [nTpB(2)/4, nTpB(2)/2, nTpB(2)*3/4]; % rest when trial index

% maxBlock = [1, 3]; 
% nTpB = [8, 8]; 
% iToR = 4;

if ~isPrac
    maxBlock = maxBlock(2);
    nTpB = nTpB(2);
else
    maxBlock = maxBlock(1);
    nTpB = nTpB(1);
end


% import stimulus
sti_info = readtable([['Rule', num2str(ruleID)],'_stimulus.csv']);
if nTpB < height(sti_info)
    indices = sort(randperm(height(sti_info), nTpB));
else
    indices = 1:height(sti_info);
end
sti_info_selected = sti_info(indices, :);

nStimulus = height(sti_info_selected);
stiID_selected = sti_info_selected.stiID;
nRep = max(nTpB/nStimulus,1);


% keys'number and category 
nOption = 2;
if isPrac || sessionID == 1
    keys_category = Shuffle(1:nOption);
else
    keys_category = load(fullfile(folders{1}, [num2str(subID),'_',num2str(sessionID-1),'_bhv.mat'])).behavior.keys_category;
end


accstop = 0.90; % stop when each category accuracy in a block reaches 90%

behavior.nTpB = nTpB;
behavior.sti_info = sti_info_selected;
behavior.nRep = nRep;
behavior.keys_category = keys_category;



%% ======================  1.3 Response key settings  =====================
KbName('UnifyKeyNames');

Keys = {'f','j'};
KeyF = KbName(Keys{1});
KeyJ = KbName(Keys{2});
opt_names = [upper(Keys{1}), upper(Keys{2})];

escape_key = KbName('ESCAPE');
space_key = KbName('SPACE');

% rating keys
rating_key_names = {'1!','2@','3#','4$','5%','6^'};
rating_keys = cellfun(@KbName, rating_key_names);
scale = 1:6;

RestrictKeysForKbCheck([KeyF, KeyJ, rating_keys, space_key, escape_key]);

%% ========================  1.4 Display settings  ========================

% select screen for display
screenID = 0;
w_rect = Screen('Rect', screenID);
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

% transform box stimulus
box_size = 60;
box_gap = 30;
box_totalwidth = 5 * box_size + 4 * box_gap;
boxstart_x = (w_width - box_totalwidth) / 2;
boxstart_y = center_y - 100;

% options (binary choice)
opt_radius = 25;
opt_width = 3;
opt_txt_size = 25;

opt_xys = [w_width*2/5-opt_radius, w_width*3/5-opt_radius; ...
           w_height*4/5-opt_radius, w_height*4/5-opt_radius; ...
           w_width*2/5+opt_radius, w_width*3/5+opt_radius; ...
           w_height*4/5+opt_radius, w_height*4/5+opt_radius];
opt_txt = opt_names;
opt_txt_xys = [w_width*2/5, w_height*4/5];

% microphone
mic_width = 0.1*w_width;
ratio = 0.75;
mic_height = mic_width/ratio;
mic_rect = [center_x-mic_width/2, center_y-mic_height/2, center_x+mic_width/2, center_y+mic_height/2];

% colors
black_val = BlackIndex(screenID);
white_val = WhiteIndex(screenID);
grey_val = white_val / 2;
bg_color = grey_val;
fix_color = [0 115 0];
opt_color = black_val;
sopt_color = [242,242,242];


%% ===================  1.5 Initialize a screen window  ===================

AssertOpenGL;

Screen('Preference', 'SkipSyncTests', 0);
Screen('Preference', 'VBLTimestampingMode', 1);
Screen('Preference', 'ConserveVRAM', 8);
Screen('Preference', 'TextEncodingLocale', 'UTF8');
Screen('Preference', 'TextRenderer', 0);

[wPtr, ~] = PsychImaging('OpenWindow', screenID, bg_color, w_rect);
Screen('TextFont', wPtr, 'simhei');

priorityLevel = MaxPriority(wPtr);
Priority(priorityLevel);

% ------ microphone symbol
[microimg, ~, microalpha] = imread('microphone.png');
microimg(:, :, 2) = microimg(:, :, 1);
microimg(:, :, 3) = microimg(:, :, 1);
microimg(:, :, 4) = microalpha;
mic_texture = Screen('MakeTexture', wPtr, microimg);


%% =========================  1.6 Introductions  ==========================

welcome_txt = '正式阶段\n\n准备好后请按【空格】键开始';
welcome_prc_txt = '练习阶段\n\n准备好后请按【空格】键开始';
rec_txt = '请按【空格】键开始录音';
rest_txt = '休息一下，倒计时%i秒';
restend_txt = '休息好后请按【空格】键继续';
blockend_txt = '你在过去一段时间的正确率为%3.3f\n\n请按【空格】键继续';
end_txt = '任务到此结束\n\n你的正确率为%3.3f';
end_prc_txt = '练习结束\n\n请向主试示意';

%% ===================  1.7 Experiment timing settings  ===================

fix_numSecs = 1;
fixj_numSecs = 0.3;
sopt_numSecs = 0.5;
prs_numSecs = 1.5;
ratemax_numSecs = 30;
blank_numSecs = 0.5;
rest_numSecs = 3;

ifi = Screen('GetFlipInterval', wPtr);

fix_fixed = round(fix_numSecs / ifi) * ifi;
fix_jitter = round(fixj_numSecs / ifi) * ifi;
sopt_duration = round(sopt_numSecs / ifi) * ifi;
prs_duration = round(prs_numSecs / ifi) * ifi;
ratemax_duration = round(ratemax_numSecs / ifi) * ifi;

%% ============================  2 Experiment  ============================

try
    PsychJavaTrouble;
    ListenChar(2);
    HideCursor(wPtr);

    %% -------------------------  2.1 Welcome  ----------------------------
    if ~isPrac
        showtext(wPtr, double(welcome_txt), font_size, black_val);
    else
        showtext(wPtr, double(welcome_prc_txt), font_size, black_val);
    end
    Screen('Flip',wPtr);
    wait_for_space_or_escape(space_key, escape_key, dataname, behavior);

    %% -------------------  2.2 Block and trial loop  ---------------------
    sti_order = [];
    trl_start = [];
    trl_end = [];
    
    category = [];
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

    % ###########################  block loop  ############################
    for iBlock = 1:maxBlock
        if finish(iBlock) == 1
            break;
        else
            % get the stimulus order of stimulus
            sti_order = cat(1, sti_order, Shuffle(repmat(stiID_selected, nRep, 1)));
            trl_start = cat(1, trl_start, nTpB*(iBlock-1)+1);
            trl_end = cat(1, trl_end, nTpB*(iBlock));

            behavior.sti_order = sti_order;
            behavior.trl_start = trl_start;
            behavior.trl_end = trl_end;
            
            save([dataname, '.mat'], 'behavior');

            % ######################  trial loop  #########################
            for iTrial = trl_start(end):trl_end(end)
                this_stiID = sti_order(iTrial);
                this_info = sti_info(sti_info.stiID==this_stiID,:);
    
                % ----- 1) show fixation
                Screen('FillOval', wPtr, fix_color, fix_position);
                Screen('DrawingFinished', wPtr);
                [~, fix_StartTime] = Screen('Flip', wPtr);
    
                fix_duration = fix_fixed + fix_jitter*rand;
                fix_EndTime = fix_StartTime + fix_duration;
                while GetSecs < fix_EndTime
                end
    
                % ----- 2) show stimuli and options
                this_features = [this_info.feature1, this_info.feature2, ...
                            this_info.feature3, this_info.feature4, this_info.feature5];
                draw_feature_squares(wPtr, this_features, boxstart_x, boxstart_y, box_size, box_gap, white_val, black_val);

                showoptions(wPtr, opt_xys, opt_width, opt_color, opt_txt, opt_txt_xys, opt_txt_size);
                Screen('DrawingFinished', wPtr);
                
                [~, sti_StartTime] = Screen('Flip', wPtr);
    
                % ----- 3) detect response
                KbReleaseWait();
                while 1
                    [keyIsDown, ~, keyCode] = KbCheck;
                    if (keyIsDown && size(find(keyCode), 2) == 1)
                        pressedKey = find(keyCode, 1);
                        if (keyCode(KeyF) || keyCode(KeyJ))
                            cho_Time = GetSecs;
                            break;
                        elseif pressedKey == escape_key
                            save([dataname, '.mat'], 'behavior');
                            sca; ListenChar(0); error('experiment aborted by user');
                        end
                    end
                end
                KbReleaseWait();
    
                % ----- 4) record response
                cho_RT = cho_Time - sti_StartTime;
                chotiming = cat(1, chotiming, [sti_StartTime, cho_Time, cho_RT]);
    
                if (ismember({KbName(keyCode)}, Keys(1)))
                    presskey = cat(1, presskey, 1);
                    choice = cat(1, choice, keys_category(1));
                elseif (ismember({KbName(keyCode)}, Keys(2)))
                    presskey = cat(1, presskey, 2);
                    choice = cat(1, choice, keys_category(2));
                end
    
                % ----- 5) show selected option
                draw_feature_squares(wPtr, this_features, boxstart_x, boxstart_y, box_size, box_gap, white_val, black_val)
                showselectedoption(wPtr, opt_xys, sopt_color, KbName(keyCode), Keys);
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
    
                % ----- 6) Confidence rating
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
                    wait_for_space_or_escape(space_key, escape_key, dataname, behavior);
        
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
        
                    Screen('FillRect', wPtr, bg_color);
                    Screen('Flip',wPtr);
                    WaitSecs(blank_numSecs);
                end

                % ----- 9) show feedback
                thiscat = this_info.category;
                category = cat(1, category, thiscat);
                if ~isPrac
                    if choice(end) == thiscat
                        thisfdb = 1;
                    else
                        thisfdb = 0;
                    end
                else
                    thisfdb = round(rand);
                end
                feedback = cat(1, feedback, thisfdb);
    
                showtext(wPtr, double(num2str(thisfdb)), 70, 0);
                Screen('Flip', wPtr);
                wait_for_space_or_escape(space_key, escape_key, dataname, behavior);
    
                % ----- 10) data recording
                behavior.category = category;
                behavior.choice = choice;
                behavior.presskey = presskey;
                behavior.rating = rating;
                behavior.feedback = feedback;
                behavior.chotiming = chotiming;
                behavior.ratetiming = ratetiming;
                if IsRecording
                    behavior.rectiming = rectiming;
                end
    
                fprintf(fid, '%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%3.3f,%3.3f,%3.3f\n', ...
                    ruleID, sessionID, iBlock, iTrial, this_stiID, thiscat, ...
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
                        wait_for_space_or_escape(space_key, escape_key, dataname, behavior);
                    end
                end
            end
            % ####################  trial loop end  #######################

            % #################  judge, generate, and rest  ###############
            if ~isPrac
                % calculate and judge whether finish
                thisacc = sum(feedback(trl_start(end):trl_end(end)))/nTpB;
                accuracy = cat(1, accuracy, thisacc);
                behavior.accuracy = accuracy;

                save([dataname, '.mat'], 'behavior');

                showtext(wPtr, double(sprintf(blockend_txt, thisacc)), font_size, 0);
                Screen('Flip', wPtr);
                wait_for_space_or_escape(space_key, escape_key, dataname, behavior);


                if thisacc > accstop
                    finish(iBlock+1) = 1;
                else
                    finish(iBlock+1) = 0;
                end

                if finish(iBlock+1) == 1 || iBlock == maxBlock
                    break;
                else
                    % rest before next block
                    ts = Screen('Flip', wPtr);
                    for iTime = 1:rest_numSecs
                        % show countdown
                        showtext(wPtr, double(sprintf(rest_txt, rest_numSecs+1-iTime)), font_size, 0);
                        ts = Screen('Flip', wPtr, ts+1);
                    end
                    WaitSecs(1);
                    Screen('Flip', wPtr, ts+1);

                    % ask if ends rest
                    showtext(wPtr, double(restend_txt), font_size, 0);
                    Screen('Flip', wPtr);
                    wait_for_space_or_escape(space_key, escape_key, dataname, behavior);
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
%%
function showtext(w, text, size, color)
Screen('TextSize', w, size);
DrawFormattedText(w, text, 'center', 'center', color);
end

%%
function draw_feature_squares(w, features, boxstart_x, boxstart_y, box_size, box_gap, white_val, black_val)
for i = 1:length(features)
    if features(i) == 1 % +1 = white, -1 = black
        color = white_val;
    elseif features(i) == -1
        color = black_val;
    end
    
    x_left = boxstart_x + (i-1) * (box_size + box_gap);
    y_top = boxstart_y;
    x_right = x_left + box_size;
    y_bottom = y_top + box_size;

    Screen('FillRect', w, color, [x_left, y_top, x_right, y_bottom]);
    Screen('FrameRect', w, 0, [x_left, y_top, x_right, y_bottom], 2);
end
end

%%
function showoptions(w, opt_xys, opt_width, opt_color, opt_txt, opt_txt_xys, opt_txt_size)
Screen('FrameOval', w, opt_color, opt_xys, opt_width);

Screen('TextSize', w, opt_txt_size);
textBounds = Screen('TextBounds', w, char(opt_txt(1)));
txt_x = opt_txt_xys(1) - textBounds(3) / 2;
txt_y = opt_txt_xys(2) - textBounds(4) / 2;
interval = opt_xys(1,2) - opt_xys(1,1);

for iKey = 1:numel(opt_txt)
    Screen('DrawText', w, char(opt_txt(iKey)), txt_x + (iKey-1)*interval, txt_y);
end
end

%%
function showselectedoption(w, opt_xys, sopt_color, keyPress, Keys)
if ismember({keyPress}, Keys(1))  % press the F key
    Screen('FillOval', w, sopt_color, opt_xys(:,1));
elseif ismember({keyPress}, Keys(2)) % press the J key
    Screen('FillOval', w, sopt_color, opt_xys(:,2));
end
end

%%
function wait_for_space_or_escape(space_key, escape_key, dataname, behavior)
KbReleaseWait();
while 1
    [keyIsDown, ~, keyCode] = KbCheck;
    if keyIsDown && nnz(keyCode) == 1
        pressedKey = find(keyCode, 1);
        if pressedKey == space_key
            break;
        elseif pressedKey == escape_key
            save([dataname, '.mat'], 'behavior');
            sca; ListenChar(0); error('experiment aborted by user');
        end
    end
end
KbReleaseWait();
end
