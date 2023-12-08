## 简介
AI-ToolKit 是一个基于PyTorch的开源工具箱。<br>
主分支代码目前支持 PyTorch 1.8 及其以上的版本。

## 最新进展
TODO

## 教程
TODO

## 项目库
<div align="center">
  <b>比赛代码</b>
</div>
<table align="center">
    <tbody>
        <tr align="center" valign="bottom">
            <th>
                <b>比赛名称</b>
            </th>
            <th>
                <b>模型架构</b>
            </th>
            <th>
                <b>比赛名次</b>
            </th>
        </tr>
        <tr>
            <td>
                <p>2022 中国高校计算机大赛-大数据挑战赛</p>
            </td>
            <td>
                <p>simple transformer</p>
            </td>
            <td>
                <p>全国三等奖</p>
            </td>
        </tr>
        <tr>
            <td>
                <p>2023 IJCAI Workshop Challenge Track 1: Horizontal Detection </p>
            </td>
            <td>
                <p>GFL w/ AFFDN</p>
            </td>
            <td>
                <p> 23th in set1</p>
            </td>
        </tr>
    </tbody>
</table>

<div align="center">
  <b>论文复现</b>
</div>
<table align="center">
    <tbody>
        <tr align="center" valign="bottom">
            <th>
                <b>论文名称</b>
            </th>
            <th>
                <b>复现实现</b>
            </th>
            <th>
                <b>复现配置</b>
            </th>
            <th>
                <b>复现结果</b>
            </th>
        </tr>
        <tr>
            <td>
                <li><a href="https://arxiv.org/abs/2302.06675">Symbolic Discovery of Optimization Algorithms</a></li>
            </td>
            <td>
                <li><a href="core/engine/optimizers/lion.py">Lion</a></li>
            </td>
            <td>
                <p>待定</p>
            </td>
            <td>
                <p>待定</p>
            </td>
        </tr>
    </tbody>
</table>


<div align="center">
  <b>眼花缭乱的Trick</b>
</div>
<table align="center">
    <tbody>
        <tr align="center" valign="bottom">
            <th>
                <b>数据处理</b>
            </th>
            <th>
                <b>模块</b>
            </th>
            <th>
                <b>训练</b>
            </th>
            <th>
                <b>loss</b>
            </th>
            <th>
                <b>后处理</b>
            </th>
        </tr>
        <tr valign="top">
        <td>
            <ul>
                <li><a href="projects/method/特征处理/demo.ipynb">特征平滑</a></li>
            </ul>
            <ul>
                <li><a href="projects/method/特征处理/demo.ipynb">log变换</a></li>
            </ul>
            <ul>
                <li><a href="projects/method/特征筛选/demo.ipynb">对抗验证</a></li>
            </ul>
            <ul>
                <li><a href="projects/method/特征筛选/demo.ipynb">Null Importance</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="core/models/attentions/dropkey.py">DropKey</a></li>
            </ul>
            <ul>
                <li><a href="core/models/attentions/simam.py">SimAM</a></li>
            </ul>
            <ul>
                <li><a href="core/models/attentions/triplet.py">Triplet Attention</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="projects/method/LGB直接优化QWK">LGB直接优化QWK</a></li>
            </ul>
            <ul>
                <li><a href="core/engine/hooks/adv">对抗训练</a></li>
            </ul>
            <ul>
                <li><a href="core/engine/optimizers/lion.py">Lion</a></li>
            </ul>
            <ul>
                <li><a href="core/engine/optimizers/gc_adamw.py">梯度中心化</a></li>
            </ul>
            <ul>
                <li>R-Drop</li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="core/evalution/functional/mask_loss.py">Mask Loss</a></li>
            </ul>
            <ul>
                <li>ArcFace Loss</li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="core/evalution/functional/">投票法</a></li>
            </ul>
        </td>
    </tr>
    </tbody>
</table>

## 可视化
<table align="center">
    <tbody>
        <tr align="center" valign="bottom">
            <th>
                <b>方法</b>
            </th>
            <th>
                <b>示例</b>
            </th>
            <th>
                <b>结果展示</b>
            </th>
        </tr>
        <tr>
            <td>
                <li><a href="tools/visualizations/gradcam">GradCAM</a></li>
            </td>
            <td>
                <li><a href="demo/demo_gradcam.py">Demo</a></li>
            </td>
            <td>
                <p>待定</p>
            </td>
        </tr>
    </tbody>
</table>

## 常见问题

TODO