# 修正后的历史结果

本目录保留原实验的预测选择和搜索轨迹，仅将真实标签的强制序号名次精确映射成并列最小名次。有效tau原始数值未变；step5中728条原先由未定义值转换为0的记录已恢复NaN并标记状态（720条通过常量特征证明，8条经定向重跑确认）。metric_version=min-rank-legacy-selection-v1；不是稳定预测并列策略的全量重跑。

- migration_changes.csv：各文件变更条数与最大名次修正。
- migration_manifest.json：原始结果文件哈希与迁移方法。
- step6/success_comparison.csv：原始与修正后找到最优比例。
- group_mean_references.csv：按各数据池计算的描述性oracle参考，非理论上界。
- step3/agreement_stratified.csv：按样本数和任务/格式分层的相关性及偏差。
- step3/paired_effects.csv：共同种子的配对差值、条件于当前结构池的区间及保守Bonferroni校正。
- smoke/validation.json：8次真实训练验证和预算/缓存检查。

原结果里的收敛警告没有记录，无法事后恢复；所有已保存tau=0已逐项处理，见undefined_tau_repair.json。新实验保留tau_raw、tau_status及warnings，默认输出results_v3。所有词频是Pythia分词器共同基线，不能称为Qwen自身token计数。
