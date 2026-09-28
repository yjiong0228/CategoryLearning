%% ==========================  1 Task settings  ===========================

% ------ stimulus settings
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

% ------ task settings
nCategory = 4; % number of categories(species)
nFeature = 4; % number of features in total
nFtSelect = 3; % number of features selected
nSelect = nchoosek(nFeature, nFtSelect); % number of feature selection (3 out of 4) methods
nSegmt = size(perms(1:nFtSelect),1); % number of feature space segmentation methods

nSub = 24;
version = repelem(1,nSub); % 1 = learn length; 2 = learn angle
condition = repmat([1,2,3],1,nSub/3); % 1 = only learn family; 2 = only learn species; 3 = learn both

nCpO = [2, 1]; % number of categories(species) per option
nOption = nCategory./nCpO; % number of options, separately for condition 1 and condition 2/3

rng(20240922);
structure = [Shuffle(repelem(1:nSelect,nSegmt)'), Shuffle(repelem(1:nSegmt,nSelect)')];
writematrix(structure, 'structure.csv');
structure = readmatrix('structure.csv'); % category structure, with format of [feature selection, feature space segmentation]

% ------ procedure settings
if strcmp(taskID, 'Task2')
    % the percent of ambigous stimulus when sampling
    ampercent = 0.1; 
    
    if debug_mode
        maxBlock = [1, 3]; % maximum block number, separately for practice and formal task
        nTpB = [8, 8]; % trial number per block, separately for practice and formal task
        iToR = 4; % rest when trial index
    else
        maxBlock = [1, 5];
        nTpB = [8, 64];
        iToR = [nTpB(2)/4, nTpB(2)/2, nTpB(2)*3/4]; % rest when trial index
    end
    
    beta = 0.2; % parameter to control the generation for the next block
    accstop = 0.9; % stop learning when accuracy meets criteria

    % recording


end


if strcmp(taskID, 'Task2') || strcmp(taskID, 'Task3a') || strcmp(taskID, 'Task3b')
    %% =========================  2 Display settings  =========================
    % ------ general display settings
    % select screen for display
    screens = Screen('Screens');
    screenID = max(screens);
    rect = Screen('Rect', screenID); % the size of the window in display pixels
    
    % basic colors
    black_val = BlackIndex(screenID);
    white_val = WhiteIndex(screenID);
    grey_val = white_val / 2;
    
    bg_color = grey_val;
    txt_color = black_val;
    
    % window size
    if debug_mode
        sz = rect*0.5;
    else
        sz = rect;
    end
    w_width = sz(3);
    w_height = sz(4);
    center_x = w_width / 2;
    center_y = w_height / 2;
    
    % visual angle
    visual_angle = 5;
    distance = 60;
    
    % -------- settings for each step
    % 1) fixation
    fix_width = 0.01*w_width;
    fix_size = 6;
    fix_xys = [center_x, center_x, center_x-fix_width, center_x+fix_width; ...
        center_y-fix_width, center_y+fix_width, center_y, center_y];
    
    % 2) stimuli figures
    ratio = 0.75; % width-height ratio of stimuli figure
    sti_width = 0.4*w_width; % proportion of figure to the screen
    sti_height = sti_width*ratio;
    if strcmp(taskID, 'Task2') || strcmp(taskID, 'Task3a')
        sti_rect = [center_x-sti_width/2, center_y-sti_height/2, center_x+sti_width/2, center_y+sti_height/2];
    elseif strcmp(taskID, 'Task3b')
        distance = 0.4*w_width; % distance of left and right animal's midpoint of body
        left_rect = [center_x-distance/2-sti_width/2, center_y-sti_height/2, center_x-distance/2+sti_width/2, center_y+sti_height/2];
        right_rect = [center_x+distance/2-sti_width/2, center_y-sti_height/2, center_x+distance/2+sti_width/2, center_y+sti_height/2];
    end
    % drawing settings
    dot_size = 75;
    line_width = 6;
    
    % 3) options
    opt_radius = 25;
    opt_width = 3;
    opt_txt_size = 25;
    opt_color = black_val;
    sopt_color = [242,242,242]; % higlighed selected option
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
    
    % 4) microphone
    mic_width = 0.1*w_width; % proportion of figure to the screen
    mic_height = mic_width/ratio;
    mic_rect = [center_x-mic_width/2, center_y-mic_height/2, center_x+mic_width/2, center_y+mic_height/2];
     
    
    %% =======================  3 Experiment settings  ========================
    % response key settings
    if strcmp(taskID, 'Task2') || strcmp(taskID, 'Task3a')
        Keys = {'f','g','h','j'};
        Key1 = KbName(Keys{1});
        Key2 = KbName(Keys{2});
        Key3 = KbName(Keys{3});
        Key4 = KbName(Keys{4});
        opt_names = [sprintf('%s',(upper(Keys{1}))),...
            sprintf('%s',(upper(Keys{2}))),...
            sprintf('%s',(upper(Keys{3}))),...
            sprintf('%s',(upper(Keys{4})))];
    elseif strcmp(taskID, 'Task3b')
        Keys = {'f','j'};
        Key1 = KbName(Keys{1});
        Key2 = KbName(Keys{2});
        opt_names = ['是','否'];
    end
    
    escapeKey = KbName('q');
    spaceKey = KbName('SPACE');
    
    % experiment timing settings
    if debug_mode
        fix_numSecs = 0.1; 
        fixj_numSecs = 0;
        sopt_numSecs = 0.1;
        prs_numSecs = 0.1;
        fdb_numSecs = 0.1;
        blank_numSecs = 0.1;
        rest_numSecs = 3;
    else
        fix_numSecs = 1; 
        fixj_numSecs = 1;
        sopt_numSecs = 1;
        prs_numSecs = 2;
        fdb_numSecs = 2;
        blank_numSecs = 1;
        rest_numSecs = 30;
    end
    
    % instructions
    if strcmp(taskID, 'Task2')
        welcome_txt = '任务2正式阶段 \n\n\n 准备好后请按【空格】键开始';
        welcome_prc_txt = '任务2练习阶段 \n\n\n 准备好后请按【空格】键开始';
        rec_txt = '请按【空格】键开始录音';
        rest_txt = '休息一下，倒计时%i秒';
        restend_txt = '休息好后请按【空格】键继续';
        end_txt = '任务2到此结束! \n\n\n 您一共获得了%3.2f元报酬。';
        end_prc_txt = '练习结束! \n\n\n 请向主试示意。';
    elseif strcmp(taskID, 'Task3a')
        welcome_txt = '任务3a \n\n\n 准备好后请按【空格】键开始';
        rest_txt = '休息一下，倒计时%i秒';
        restend_txt = '休息好后请按【空格】键继续';
        end_txt = '任务3a到此结束! \n\n\n 您一共获得了%3.2f元报酬。';
    elseif strcmp(taskID, 'Task3b')
        welcome_txt = '任务3b正式阶段 \n\n\n 准备好后请按【空格】键开始';
        welcome_prc_txt = '任务3b练习阶段 \n\n\n 准备好后请按【空格】键开始';
        rest_txt = '休息一下，倒计时%i秒';
        restend_txt = '休息好后请按【空格】键继续';
        end_txt = '任务3b到此结束! \n\n\n 您一共获得了%3.2f元报酬。';
        end_prc_txt = '练习结束! \n\n\n 请向主试示意。';
    end
end
