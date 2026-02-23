# Maxwell Displacement Gating PINN

基于 Maxwell 方程的位移门控物理神经网络实验（1D / 2D）。

## 项目介绍

这是我个人从零学习 AI 和物理时做的一个练习项目。  
我本身不是博士，也不是学术圈的人，之前主要是在汽车行业做生意。  
因为对 AI 感兴趣，我一步一步学代码、学 Maxwell 方程，做出了这个位移门控（DispField）的实验。

这个项目里有：

- 1D / 2D Maxwell PINN 的训练代码  
- Baseline 和带位移门控（DispField）的模型  
- 一份 `REPORT.md`，总结了 L2 误差、稀疏度和推荐超参数  
- 一份 `paper_draft.md`，是准备投稿用的草稿  

这个仓库的目标是留下一个真实的学习过程：  
我会把现在能跑通、验证过的结果都记录下来，给未来的自己和别人参考。

## 环境需求

- Python 3.10+
- PyTorch（支持 CUDA）
- 一张支持 CUDA 的显卡（例如 RTX 3050）

## 安装步骤

```bash
git clone https://github.com/jordenlcc-bot/MaxWell-.git
cd MaxWell-
pip install -r requirements.txt
运行 1D / 2D 实验
运行 1D 实验：

bash
python run_experiment_1d.py
运行 2D 实验：

bash
python run_experiment_2d.py
运行完成后，可以在 REPORT.md 中查看 1D / 2D 的 L2 误差和稀疏度结果。

注意：  
- 不要把“很好，排版已经差不多了”这种聊天文字一起贴进去。  
- 不要多加别的东西，先完全照着这块来。
