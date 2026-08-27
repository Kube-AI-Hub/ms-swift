// SWIFT Web-UI Restful i18n
window.i18n = {
  zh: {
    // Header
    title: 'SWIFT: 轻量级大模型训练推理框架',
    subtitle: '请查看 <a href="https://github.com/modelscope/ms-swift/tree/main/docs/source" target="_blank">SWIFT 文档</a>来查看更多功能',

    // Top tabs
    tabTrain: 'LLM预训练/微调',
    tabRlhf: 'LLM人类对齐',
    tabGrpo: 'LLM GRPO',
    tabInfer: 'LLM推理',
    tabExport: 'LLM导出',
    tabEval: 'LLM评测',
    tabSample: 'LLM采样',

    // Common fields
    fieldModel: '模型路径/ID',
    fieldModelType: '模型类型',
    fieldTemplate: '对话模板',
    fieldDataset: '数据集',
    fieldTunerType: '训练方式',
    fieldSeed: '随机数种子',
    fieldTorchDtype: '训练精度',
    fieldUseLigerKernel: '使用 Liger Kernel',
    fieldGpuIds: '选择 GPU',
    fieldNpuIds: '选择 NPU',
    fieldMluIds: '选择 MLU',
    gpuKindGpu: 'GPU',
    gpuKindNpu: 'NPU',
    gpuKindMlu: 'MLU',
    gpuKindMps: 'MPS',
    gpuKindCpu: 'CPU',
    fieldUseDdp: '使用 DDP',
    fieldDdpNum: 'DDP 分片数',
    fieldDeepspeed: 'DeepSpeed',
    fieldSeqParallel: '序列并行大小',
    fieldLearningRate: '学习率',
    fieldBatchSize: '每卡训练 Batch Size',
    fieldEpochs: '训练轮数',
    fieldOutputDir: '输出目录',
    fieldLoggingDir: '日志目录',
    fieldSystem: 'System Prompt',
    fieldEnvs: '额外环境变量',
    fieldDryRun: '仅生成命令（不执行）',
    fieldMoreParams: '其他参数（JSON 或 --key value）',

    // Section labels
    sectionTrainParams: '训练参数',
    sectionChat: '对话',

    errMissingModel:   '请填写模型路径/ID',
    errMissingDataset: '请选择或填入一个数据集',
    warnModelMetaUnknown: '无法自动识别该模型的 model_type/template，请手动确认',
    infoModelMetaFromArgs: '已从本地 args.json 恢复模型参数',
    successKillTask:   '任务已成功终止',
    loadingLog:        '正在加载日志...',

    // Train accordion section labels
    sectionBasicConfig:   '基础配置',
    sectionDatasetConfig: '数据集设置',
    sectionLoraParams:    'LoRA 参数',
    sectionHardware:      '硬件与加速',
    sectionHardwareAccel: '硬件与加速',
    sectionAdvancedTrain: '高级训练参数',
    sectionAdvancedConfig: '高级参数',
    sectionAlignmentConfig: '对齐参数',
    sectionRewardConfig: '奖励函数',
    sectionRolloutConfig: 'Rollout / vLLM',
    sectionOtherReport:   '其他 & 上报',

    // Train specific
    fieldTrainStage: '训练阶段',
    fieldTrainRecord: '训练记录',
    hintTrainRecord: '选择历史训练记录可恢复参数',
    optNoRecord: '-- 无记录 --',
    btnClearRecords: '删除记录',

    // Dataset settings
    fieldSplitRatio: '验证集拆分比例',
    hintSplitRatio: '拆分到验证集的比例，0 表示不拆分',
    fieldMaxLength: '句子最大长度',
    hintMaxLength: '输入模型的最大 token 长度',
    fieldPaddingFree: '无填充批处理',

    // Hyper params
    fieldEvalBatchSize: '每卡验证 Batch Size',
    fieldGradAccum: '梯度累积步数',
    fieldEvalSteps: '验证步数',
    fieldSaveSteps: '保存步数',
    fieldAttnImpl: 'Flash Attention 类型',
    fieldNeftune: 'NEFTune 噪声系数',
    hintNeftune: '通常设为 5 或 10',
    hintLoggingDir: '为空则自动生成',

    // Advanced accordion
    sectionAdvancedParams: '高级参数',
    tabAdvanced: '高级设置',
    tabLoRA: 'LoRA 参数',
    tabTask: '任务类型',
    tabSelfCog: '自我认知',
    tabHub: 'Hub 保存',
    tabReport: '上报',
    tabOther: '其他',
    tabMoreParams: '其他参数',

    // Advanced settings
    fieldTunerBackend: 'Tuner Backend',
    fieldWeightDecay: '权重衰减',
    fieldLoggingSteps: '日志打印步数',
    fieldLrScheduler: 'LrScheduler 类型',
    fieldWarmupRatio: '学习率 Warmup 比例',
    fieldTruncation: '数据集超长策略',
    fieldMaxSteps: '最大迭代步数',
    hintMaxSteps: '大于 0 时忽略训练轮数',
    fieldMaxGradNorm: '梯度裁剪',

    // LoRA
    fieldLoraRank: 'LoRA Rank',
    fieldLoraAlpha: 'LoRA Alpha',
    fieldLoraDropout: 'LoRA Dropout',
    fieldLoraDtype: 'LoRA Dtype',
    fieldUseRsLora: '使用 rsLoRA',
    fieldUseDoRA: '使用 DoRA',
    fieldUseRslora: '使用 rsLoRA',
    fieldUseDora: '使用 DoRA',
    fieldTargetModules: 'Target Modules',
    hintTargetModules: '逗号分隔，或 all-linear',

    // Task type
    fieldTaskType: '任务类型',
    fieldLossType: 'Loss 类型',
    fieldNumLabels: '标签数量',
    fieldUseChatTemplate: '使用 Chat Template',

    // Self-cognition
    fieldModelName: '模型名称',
    hintModelName: '自我认知微调的模型名称',
    fieldModelAuthor: '模型作者',
    hintModelAuthor: '自我认知微调的模型作者',

    // Hub save
    fieldPushToHub: '推送到 Hub',
    fieldHubModelId: 'Hub 模型 ID',
    fieldHubPrivate: '私有仓库',
    fieldHubStrategy: '推送策略',
    fieldHubToken: 'Hub Token',

    // Reporting
    fieldReportTo: '上报平台',
    fieldSwanlabToken: 'SwanLab Token',
    fieldSwanlabProject: 'SwanLab Project',
    fieldSwanlabWorkspace: 'SwanLab Workspace',
    fieldSwanlabExpName: 'SwanLab 实验名称',
    fieldSwanlabMode: 'SwanLab Mode',

    // RLHF specific
    fieldRlhfType: 'RLHF 类型',
    fieldRefModel: '参考模型',
    fieldRefModelType: '参考模型类型',
    fieldRewardModel: '奖励模型',
    fieldRewardModelType: '奖励模型类型',
    fieldTeacherModel: '教师模型',
    fieldTeacherModelType: '教师模型类型',
    fieldBeta: 'Beta 系数',
    fieldMaxCompletionLength: '最大生成长度',
    fieldLossScale: '损失权重设置',
    fieldLambda: 'Lambda',
    fieldRpoAlpha: 'RPO Alpha',
    fieldCpoAlpha: 'CPO Alpha',
    fieldSimpoGamma: 'SimPO Gamma',
    fieldDesirableWeight: 'Desirable Weight',
    fieldUndesirableWeight: 'Undesirable Weight',

    // GRPO specific
    fieldVllmMode: 'vLLM 模式',
    fieldNumGenerations: '生成数量',
    fieldRewardFuncs: '奖励函数',
    fieldRewardWeights: '奖励函数权重',
    fieldVllmGpuMemUtil: 'GPU 显存利用率',
    fieldVllmTensorParallel: '张量并行大小',
    fieldVllmMaxModelLen: '模型支持的最大长度',
    fieldServerHost: 'vLLM 服务主机',
    fieldServerPort: 'vLLM 服务端口',
    fieldServerTimeout: '服务超时时间',
    fieldEpsilon: 'Clip 系数',
    fieldEpsilonHigh: 'Upper Clip 系数',
    fieldNumIterations: '每个批次更新次数',

    // Infer specific
    fieldAdapters: 'Adapter 路径',
    fieldInferBackend: '推理后端',
    fieldPort: '服务端口',
    fieldInferModelType: 'LoRA 模块名',
    fieldMaxNewTokens: '最大生成长度',
    fieldTemperature: '温度',
    fieldTopP: 'Top-p',
    fieldTopK: 'Top-k',
    fieldRepPenalty: '重复惩罚',

    // Export specific
    fieldMergeLora: '合并 LoRA',
    fieldQuantBits: '量化位数',
    fieldQuantMethod: '量化方法',
    fieldQuantNSamples: '量化样本数',
    fieldMaxLength: '最大长度',
    fieldDeviceMap: 'Device Map',
    sectionExportHub: '写回 CSGHub 模型库',
    fieldPushToCsghub: '推送到 CSGHub',
    fieldExistOk: '允许覆盖已有目录',
    hintExportHub: '勾选后将导出结果推送到 CSGHub。仓库不存在会自动创建；微调实例已注入 Token 时可留空。请使用新的命名空间/仓库名，不要覆盖基座模型。',
    hintExportHubModelId: '格式：命名空间/仓库名，例如 admin/qwen3-lora-sft',
    hintExportAdapters: '合并 LoRA 时填写训练产出的 checkpoint 目录',
    exportHubModelIdRequired: '推送到 CSGHub 时必须填写模型库 ID（命名空间/仓库名）',

    // Eval specific
    fieldEvalBackend: '评测后端',
    fieldEvalDataset: '评测数据集',
    fieldEvalLimit: '评测样本数限制',
    fieldCustomEvalConfig: '自定义评测配置',
    fieldEvalOutputDir: '评测输出目录',
    fieldEvalUrl: '评测 API URL',
    fieldApiKey: 'API Key',

    // Sample specific
    fieldSamplerType: '采样器类型',
    fieldSamplerEngine: '采样引擎',
    fieldNumReturnSeq: '每条返回数量',
    fieldNBestToKeep: '保留最优数量',
    fieldSamplingBatchSize: '采样 Batch Size',
    fieldSamplingBatches: '采样 Batch 数',
    fieldPrmModel: '过程奖励模型 (PRM)',
    fieldOrmModel: '结果奖励模型 (ORM)',

    // Buttons
    btnStart: '🚀 开始',
    btnStop: '⏹ 停止',
    btnPreview: '📋 预览命令',
    btnDeploy: '🚀 部署模型',
    btnSend: '🚀 发送',
    btnClearHistory: '清除对话',
    btnRefreshTasks: '刷新任务',
    btnShowLog: '展示日志',
    btnStopLog: '停止展示',
    btnTensorBoardOpen: '打开 TensorBoard',
    btnTensorBoardClose: '关闭 TensorBoard',
    btnKillTask: '终止任务',

    // Status
    statusIdle: '空闲',
    statusStarting: '启动中...',
    statusRunning: '运行中',
    statusDone: '完成',
    statusError: '错误',
    statusDryRun: '（仅预览，未执行）',

    // Log panel
    logTitle: '运行日志',
    cmdTitle: '生成命令',
    runningTasksLabel: '运行中任务',
    noTasks: '暂无运行中任务',
    awaitingLog: '日志将在任务启动后显示...',
    metricsPanelTitle: '训练指标（TensorBoard）',
    labelTrainLog: '训练日志',
    labelCmdPreview: '运行命令',
    metricsPanelHint: '选择运行中任务后自动刷新曲线。',
    tensorBoardUrlLabel: 'TensorBoard 链接',
    metricTrainLoss: 'train/loss',
    metricTrainAcc: 'train/acc',
    metricTrainLr: 'train/learning_rate',
    metricTrainReward: 'train/rewards',
    metricEvalLoss: 'eval/loss',
    metricEvalAcc: 'eval/acc',
    metricEvalReward: 'eval/rewards',
    metricLegendRaw: '原始',
    metricLegendSmoothed: '平滑',
    progressElapsed: '已用时间',
    progressEta: '预计剩余时间',
    progressSpeed: '速度',
    speedUnits: { 's/it': '秒/迭代', 'it/s': '迭代/秒', 'ms/it': '毫秒/迭代', 'min/it': '分钟/迭代', 'it/min': '迭代/分钟' },
    tensorBoardNeedLoggingDir: '请先选择运行中任务或启动训练以获取 logging_dir',
    tensorBoardStopped: 'TensorBoard 已关闭',

    // Chat
    chatPlaceholder: '请输入消息（Enter 发送，Shift+Enter 换行）...',
    chatEmpty: '对话为空，请先发送消息',
    thinkTitle: '思考过程',
    thinkStreaming: '思考中…',

    // Media upload
    mediaTabText: '文本',
    mediaTabImage: '图片',
    mediaTabVideo: '视频',
    mediaTabAudio: '音频',
    uploadImage: '点击或拖拽上传图片',
    uploadVideo: '点击或拖拽上传视频',
    uploadAudio: '点击或拖拽上传音频',

    // Hints
    hintModel: '本地路径或 ModelScope / HuggingFace 模型 ID',
    hintGpuIds: '可多选；CUDA/NPU/MLU 不可用时只能选 CPU',
    hintDdp: '使用数据并行训练，需配合多 GPU 使用',
    hintDeepspeed: '从下拉选择预设或填入 JSON 配置文件路径',
    hintMoreParams: '支持 JSON 格式或 --key value 命令行格式',
    hintEnvs: '格式：KEY1=VAL1 KEY2=VAL2',
    hintPort: '推理服务监听端口，默认 8000',
    datasetPlaceholder: '输入后按 Enter 添加，支持多选',
    datasetHint: '输入数据集名称或路径，按 Enter 确认；可添加多个',

    // Lang toggle
    langLabel: '语言',
  },
  en: {
    // Header
    title: 'SWIFT: Scalable lightWeight Infrastructure for Fine-Tuning and Inference',
    subtitle: 'Please check <a href="https://github.com/modelscope/ms-swift/tree/main/docs/source_en" target="_blank">SWIFT Documentation</a> for more usages',

    // Top tabs
    tabTrain: 'LLM PT/SFT',
    tabRlhf: 'LLM RLHF',
    tabGrpo: 'LLM GRPO',
    tabInfer: 'LLM Inference',
    tabExport: 'LLM Export',
    tabEval: 'LLM Evaluation',
    tabSample: 'LLM Sampling',

    // Common fields
    fieldModel: 'Model path / ID',
    fieldModelType: 'Model type',
    fieldTemplate: 'Chat template',
    fieldDataset: 'Dataset',
    fieldTunerType: 'Train type',
    fieldSeed: 'Seed',
    fieldTorchDtype: 'Training precision',
    fieldUseLigerKernel: 'Use Liger Kernel',
    fieldGpuIds: 'Choose GPU',
    fieldNpuIds: 'Choose NPU',
    fieldMluIds: 'Choose MLU',
    gpuKindGpu: 'GPU',
    gpuKindNpu: 'NPU',
    gpuKindMlu: 'MLU',
    gpuKindMps: 'MPS',
    gpuKindCpu: 'CPU',
    fieldUseDdp: 'Use DDP',
    fieldDdpNum: 'DDP sharding num',
    fieldDeepspeed: 'DeepSpeed',
    fieldSeqParallel: 'Sequence parallel size',
    fieldLearningRate: 'Learning rate',
    fieldBatchSize: 'Per-device train batch size',
    fieldEpochs: 'Epochs',
    fieldOutputDir: 'Output dir',
    fieldLoggingDir: 'Logging dir',
    fieldSystem: 'System prompt',
    fieldEnvs: 'Extra env vars',
    fieldDryRun: 'Dry-run (preview only)',
    fieldMoreParams: 'Other params (JSON or --key value)',

    // Section labels
    sectionTrainParams: 'Train settings',
    sectionChat: 'Chat',

    errMissingModel:   'Please enter a model path or ID',
    errMissingDataset: 'Please select or enter a dataset',
    warnModelMetaUnknown: 'Unable to infer model_type/template automatically for this model; please confirm manually',
    infoModelMetaFromArgs: 'Model parameters restored from local args.json',
    successKillTask:   'Task terminated successfully',
    loadingLog:        'Loading log...',

    // Train accordion section labels
    sectionBasicConfig:   'Basic Config',
    sectionDatasetConfig: 'Dataset Settings',
    sectionLoraParams:    'LoRA Parameters',
    sectionHardware:      'Hardware & Acceleration',
    sectionHardwareAccel: 'Hardware & Acceleration',
    sectionAdvancedTrain: 'Advanced Training',
    sectionAdvancedConfig: 'Advanced Config',
    sectionAlignmentConfig: 'Alignment Params',
    sectionRewardConfig: 'Reward Functions',
    sectionRolloutConfig: 'Rollout / vLLM',
    sectionOtherReport:   'Other & Reporting',

    // Train specific
    fieldTrainStage: 'Train stage',
    fieldTrainRecord: 'Train record',
    hintTrainRecord: 'Select a past record to restore parameters',
    optNoRecord: '-- No records --',
    btnClearRecords: 'Delete records',

    // Dataset settings
    fieldSplitRatio: 'Val split ratio',
    hintSplitRatio: 'Fraction split into validation set, 0 = no split',
    fieldMaxLength: 'Max sequence length',
    hintMaxLength: 'Maximum token length fed to the model',
    fieldPaddingFree: 'Padding-free batching',

    // Hyper params
    fieldEvalBatchSize: 'Per-device eval batch size',
    fieldGradAccum: 'Gradient accumulation steps',
    fieldEvalSteps: 'Eval steps',
    fieldSaveSteps: 'Save steps',
    fieldAttnImpl: 'Flash Attention type',
    fieldNeftune: 'NEFTune noise alpha',
    hintNeftune: 'Typically set to 5 or 10',
    hintLoggingDir: 'Auto-generated if empty',

    // Advanced accordion
    sectionAdvancedParams: 'Advanced settings',
    tabAdvanced: 'Advanced',
    tabLoRA: 'LoRA params',
    tabTask: 'Task type',
    tabSelfCog: 'Self-cognition',
    tabHub: 'Hub save',
    tabReport: 'Reporting',
    tabOther: 'Other',
    tabMoreParams: 'Other params',

    // Advanced settings
    fieldTunerBackend: 'Tuner backend',
    fieldWeightDecay: 'Weight decay',
    fieldLoggingSteps: 'Logging steps',
    fieldLrScheduler: 'LR scheduler type',
    fieldWarmupRatio: 'Warmup ratio',
    fieldTruncation: 'Truncation strategy',
    fieldMaxSteps: 'Max steps',
    hintMaxSteps: 'When > 0, num_train_epochs is ignored',
    fieldMaxGradNorm: 'Max grad norm',

    // LoRA
    fieldLoraRank: 'LoRA rank',
    fieldLoraAlpha: 'LoRA alpha',
    fieldLoraDropout: 'LoRA dropout',
    fieldLoraDtype: 'LoRA dtype',
    fieldUseRsLora: 'Use rsLoRA',
    fieldUseDoRA: 'Use DoRA',
    fieldUseRslora: 'Use rsLoRA',
    fieldUseDora: 'Use DoRA',
    fieldTargetModules: 'Target modules',
    hintTargetModules: 'Comma-separated or all-linear',

    // Task type
    fieldTaskType: 'Task type',
    fieldLossType: 'Loss type',
    fieldNumLabels: 'Number of labels',
    fieldUseChatTemplate: 'Use chat template',

    // Self-cognition
    fieldModelName: 'Model name',
    hintModelName: 'Model name for self-cognition fine-tuning',
    fieldModelAuthor: 'Model author',
    hintModelAuthor: 'Model author for self-cognition fine-tuning',

    // Hub save
    fieldPushToHub: 'Push to Hub',
    fieldHubModelId: 'Hub model ID',
    fieldHubPrivate: 'Private repo',
    fieldHubStrategy: 'Push strategy',
    fieldHubToken: 'Hub token',

    // Reporting
    fieldReportTo: 'Report to',
    fieldSwanlabToken: 'SwanLab token',
    fieldSwanlabProject: 'SwanLab project',
    fieldSwanlabWorkspace: 'SwanLab workspace',
    fieldSwanlabExpName: 'SwanLab exp name',
    fieldSwanlabMode: 'SwanLab mode',

    // RLHF specific
    fieldRlhfType: 'RLHF type',
    fieldRefModel: 'Reference model',
    fieldRefModelType: 'Reference model type',
    fieldRewardModel: 'Reward model',
    fieldRewardModelType: 'Reward model type',
    fieldTeacherModel: 'Teacher model',
    fieldTeacherModelType: 'Teacher model type',
    fieldBeta: 'Beta',
    fieldMaxCompletionLength: 'Max completion length',
    fieldLossScale: 'Loss scale',
    fieldLambda: 'Lambda',
    fieldRpoAlpha: 'RPO alpha',
    fieldCpoAlpha: 'CPO alpha',
    fieldSimpoGamma: 'SimPO gamma',
    fieldDesirableWeight: 'Desirable weight',
    fieldUndesirableWeight: 'Undesirable weight',

    // GRPO specific
    fieldVllmMode: 'vLLM mode',
    fieldNumGenerations: 'Num generations',
    fieldRewardFuncs: 'Reward functions',
    fieldRewardWeights: 'Reward weights',
    fieldVllmGpuMemUtil: 'GPU memory utilization',
    fieldVllmTensorParallel: 'Tensor parallel size',
    fieldVllmMaxModelLen: 'Max model len',
    fieldServerHost: 'vLLM server host',
    fieldServerPort: 'vLLM server port',
    fieldServerTimeout: 'Server timeout',
    fieldEpsilon: 'Clip coefficient',
    fieldEpsilonHigh: 'Upper clip coefficient',
    fieldNumIterations: 'Num iterations per batch',

    // Infer specific
    fieldAdapters: 'Adapter path',
    fieldInferBackend: 'Infer backend',
    fieldPort: 'Server port',
    fieldInferModelType: 'LoRA module name',
    fieldMaxNewTokens: 'Max new tokens',
    fieldTemperature: 'Temperature',
    fieldTopP: 'Top-p',
    fieldTopK: 'Top-k',
    fieldRepPenalty: 'Repetition penalty',

    // Export specific
    fieldMergeLora: 'Merge LoRA',
    fieldQuantBits: 'Quant bits',
    fieldQuantMethod: 'Quant method',
    fieldQuantNSamples: 'Quant n samples',
    fieldMaxLength: 'Max length',
    fieldDeviceMap: 'Device map',
    sectionExportHub: 'Push to CSGHub',
    fieldPushToCsghub: 'Push to CSGHub',
    fieldExistOk: 'Overwrite existing output dir',
    hintExportHub: 'Push the export result to CSGHub. Missing repos are created automatically. Leave the token empty if the finetune instance already injected one. Use a new namespace/repo name; do not overwrite the base model.',
    hintExportHubModelId: 'Format: namespace/repo, e.g. admin/qwen3-lora-sft',
    hintExportAdapters: 'Checkpoint directory from training, required when merging LoRA',
    exportHubModelIdRequired: 'hub_model_id (namespace/repo) is required when pushing to CSGHub',

    // Eval specific
    fieldEvalBackend: 'Eval backend',
    fieldEvalDataset: 'Eval dataset',
    fieldEvalLimit: 'Eval limit',
    fieldCustomEvalConfig: 'Custom eval config',
    fieldEvalOutputDir: 'Eval output dir',
    fieldEvalUrl: 'Eval API URL',
    fieldApiKey: 'API key',

    // Sample specific
    fieldSamplerType: 'Sampler type',
    fieldSamplerEngine: 'Sampler engine',
    fieldNumReturnSeq: 'Num return sequences',
    fieldNBestToKeep: 'N-Best to Keep',
    fieldSamplingBatchSize: 'Sampling batch size',
    fieldSamplingBatches: 'Num sampling batches',
    fieldPrmModel: 'Process Reward Model (PRM)',
    fieldOrmModel: 'Outcome Reward Model (ORM)',

    // Buttons
    btnStart: '🚀 Start',
    btnStop: '⏹ Stop',
    btnPreview: '📋 Preview',
    btnDeploy: '🚀 Deploy',
    btnSend: '🚀 Send',
    btnClearHistory: 'Clear history',
    btnRefreshTasks: 'Refresh tasks',
    btnShowLog: 'Show log',
    btnStopLog: 'Stop log',
    btnTensorBoardOpen: 'Open TensorBoard',
    btnTensorBoardClose: 'Close TensorBoard',
    btnKillTask: 'Kill task',

    // Status
    statusIdle: 'Idle',
    statusStarting: 'Starting...',
    statusRunning: 'Running',
    statusDone: 'Done',
    statusError: 'Error',
    statusDryRun: '(dry-run, not executed)',

    // Log panel
    logTitle: 'Runtime Log',
    cmdTitle: 'Command',
    runningTasksLabel: 'Running tasks',
    noTasks: 'No running tasks',
    awaitingLog: 'Log will appear after task starts...',
    metricsPanelTitle: 'Training metrics (TensorBoard)',
    labelTrainLog: 'Training log',
    labelCmdPreview: 'Run command',
    metricsPanelHint: 'Charts refresh after you select a running task.',
    tensorBoardUrlLabel: 'TensorBoard URL',
    metricTrainLoss: 'train/loss',
    metricTrainAcc: 'train/acc',
    metricTrainLr: 'train/learning_rate',
    metricTrainReward: 'train/rewards',
    metricEvalLoss: 'eval/loss',
    metricEvalAcc: 'eval/acc',
    metricEvalReward: 'eval/rewards',
    metricLegendRaw: 'raw',
    metricLegendSmoothed: 'smoothed',
    progressElapsed: 'Elapsed',
    progressEta: 'ETA',
    progressSpeed: 'Speed',
    speedUnits: { 's/it': 's/it', 'it/s': 'it/s', 'ms/it': 'ms/it', 'min/it': 'min/it', 'it/min': 'it/min' },
    tensorBoardNeedLoggingDir: 'Select a running task or start training to get logging_dir',
    tensorBoardStopped: 'TensorBoard stopped',

    // Chat
    chatPlaceholder: 'Type a message (Enter to send, Shift+Enter for newline)...',
    chatEmpty: 'No messages yet',
    thinkTitle: 'Thinking process',
    thinkStreaming: 'Thinking…',

    // Media upload
    mediaTabText: 'Text',
    mediaTabImage: 'Image',
    mediaTabVideo: 'Video',
    mediaTabAudio: 'Audio',
    uploadImage: 'Click or drag to upload image',
    uploadVideo: 'Click or drag to upload video',
    uploadAudio: 'Click or drag to upload audio',

    // Hints
    hintModel: 'Local path or ModelScope / HuggingFace model ID',
    hintGpuIds: 'Multi-select; fall back to CPU if CUDA/NPU/MLU unavailable',
    hintDdp: 'Distributed Data Parallel; requires multiple GPUs',
    hintDeepspeed: 'Select a preset or enter a path to a JSON config',
    hintMoreParams: 'JSON format or --key value CLI format',
    hintEnvs: 'Format: KEY1=VAL1 KEY2=VAL2',
    hintPort: 'Inference server port, default 8000',
    datasetPlaceholder: 'Type and press Enter to add, supports multiple values',
    datasetHint: 'Enter dataset name or path, press Enter to confirm',

    // Lang toggle
    langLabel: 'Language',
  },
};

window.applyI18n = function(lang) {
  const t = window.i18n[lang];
  if (!t) return;

  // Title (preserve version badge if already injected)
  const titleEl = document.getElementById('header-title');
  if (titleEl) {
    const badge = titleEl.querySelector('#header-version');
    titleEl.textContent = t.title;
    if (badge) titleEl.appendChild(badge);
  }

  // Subtitle and star-beggar use innerHTML (contain links)
  const subtitleEl = document.getElementById('header-subtitle');
  if (subtitleEl) subtitleEl.innerHTML = t.subtitle;

  // Top tab labels
  const tabMap = {
    'tab-train':  t.tabTrain,
    'tab-rlhf':   t.tabRlhf,
    'tab-grpo':   t.tabGrpo,
    'tab-infer':  t.tabInfer,
    'tab-export': t.tabExport,
    'tab-eval':   t.tabEval,
    'tab-sample': t.tabSample,
  };
  Object.entries(tabMap).forEach(([id, text]) => {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  });

  // All data-i18n elements
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (t[key] !== undefined) {
      if (el.getAttribute('data-i18n-attr') === 'placeholder') {
        el.placeholder = t[key];
      } else if (el.getAttribute('data-i18n-attr') === 'innerHTML') {
        el.innerHTML = t[key];
      } else {
        el.textContent = t[key];
      }
    }
  });
  if (typeof window.relabelGpuDeviceOptions === 'function') {
    window.relabelGpuDeviceOptions();
  }
};
