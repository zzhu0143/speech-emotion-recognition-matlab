%% Simple Test Script
% Test if everything works

fprintf('\n\n');
fprintf('========================================\n');
fprintf('测试脚本开始运行！\n');
fprintf('========================================\n\n');

% Check if data exists
if exist('data/RAVDESS', 'dir')
    fprintf('✓ 数据文件夹存在\n');

    % Count actors
    actors = dir('data/RAVDESS/Actor_*');
    fprintf('✓ 找到 %d 个Actor文件夹\n', length(actors));

    % Count files
    totalFiles = 0;
    for i = 1:length(actors)
        files = dir(fullfile('data/RAVDESS', actors(i).name, '*.wav'));
        totalFiles = totalFiles + length(files);
    end
    fprintf('✓ 总共 %d 个音频文件\n', totalFiles);

    if length(actors) == 24 && totalFiles > 1400
        fprintf('\n========================================\n');
        fprintf('数据集完美！可以开始训练！\n');
        fprintf('========================================\n\n');
        fprintf('请运行: main_train_all_models\n');
    else
        fprintf('\n✗ 数据集不完整\n');
    end
else
    fprintf('✗ 找不到数据文件夹\n');
end

fprintf('\n测试完成！\n');
