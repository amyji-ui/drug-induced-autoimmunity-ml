# 集群环境

项目目录：`/net/dali/home/chikina/guanyanw/DRUG_CL/drug_cl`。
上游仓库：https://github.com/amyji-ui/drug-induced-autoimmunity-ml

本地环境使用 Python 3.13.18，依赖版本来自上游 `requirements.txt`。
集群默认 Python 3.11 无法安装其中的 NumPy 2.5.2；请使用项目内的解释器。
`.python/` 保存 Python 运行时，`.tools/` 保存 uv，`.venv/` 保存项目依赖。

## 使用现有环境

先用 `squeue -u "$USER"` 确认分配，再进入相应作业。配置时使用的是
`n201` 上的作业 `57864754`；以后请替换为仍有效的作业 ID。

```bash
/opt/slurm/bin/srun --jobid=<JOB_ID> --overlap --pty bash
cd /net/dali/home/chikina/guanyanw/DRUG_CL/drug_cl
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
python util/check_environment.py
```

检查程序导入所有依赖和公共模块、读取描述符字典，在训练集上生成 scaffold 分组，
验证各折没有组重叠，并完成默认 kNN 的五次训练和预测。它不读取测试集，
不写入正式模型结果。安装验证记录位于 `logs/environment_check.log`。

科学参数和数据路径由 `config.yaml` 控制。上游提供了公共预处理、分组和调参模块，
本地现已新增 Logistic Regression 训练入口。`00_preprocessing.py`、`01_scaffold_grouping.py`
和 `02_tuning.py` 是供其他脚本导入的模块，直接执行不会启动完整训练。

## Logistic Regression

在已分配的计算节点、项目根目录运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u util/03_logistic_regression.py
```

参数网格位于 `config.yaml` 的 `models.logistic_regression`，默认执行 14 组参数 × 5 折。
程序将收敛警告视为错误；成功后保存全部逐折结果、各组合均值/标准差、最佳配置的五个模型，
并写出 `logs/tuning_logistic_regression.log`。重复运行会更新对应模型的 CSV、模型文件和日志。
本次结果说明位于 `results/logistic_regression_report.md`，是本次运行的静态记录。

共享折分配保存在 `results/cv_folds.json`，后续所有使用公共分组模块的模型会读取同一划分。
如果训练数据内容/顺序、种子或折数发生变化，程序会拒绝套用旧划分；有意进行新实验时，
请在 `config.yaml` 中设置新的 `paths.cv_folds_file`，保留原实验记录。

## 重建环境

在计算节点、项目根目录执行。以下命令适用于新环境；已经存在 `.venv` 时不必重建。

```bash
python3 -m pip install --target .tools uv
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export UV_CACHE_DIR="/tmp/$USER-drug-cl-uv"
export UV_LINK_MODE=copy
.tools/bin/uv python install 3.13.18
.tools/bin/uv venv --python 3.13.18 .venv
.tools/bin/uv pip install --python .venv/bin/python -r requirements-linux.lock.txt
.tools/bin/uv pip check --python .venv/bin/python
```

`requirements-linux.lock.txt` 记录本次 Linux 环境中的直接和传递依赖；
如需仅按上游的直接依赖解析，可将它替换为 `requirements.txt`。
uv 缓存位于节点 `/tmp`，环境位于共享文件系统，所以使用 `UV_LINK_MODE=copy`。

## VS Code

通过登录节点跳转，直接连接已分配的计算节点，仅打开本项目目录。
项目内 `.vscode/settings.json` 已选择 `.venv/bin/python`，关闭 Pylance 索引，
并排除环境目录和生成文件的监听与搜索。SSH 跳转和服务端 `/tmp` 安装路径
需在连接集群的客户端配置。

完整安装过程、问题及解决记录见 `PROGRESS.md`。
