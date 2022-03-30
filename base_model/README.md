## 神经网络Baseline基本模型

------

### 目录结构

---

    --root_dir
    	--data_batching(训练数据)
    		--raw_data(原始数据)
    		--clean_data(清洗后数据)
    	--data_toolkit(数据处理工具)
    		--hyperparameters_manager.py(超参管理)
    	--result(模型训练结果)
    		--plots（loss画图）
    		--saved_model(最优模型)
    	--training(训练)
    		--baseic_model.py(模型类)
    		--data_process.py(预处理)
    		--train.py(训练类)
    		--baseline_inference.py(baseline推理)
    		--run.sh(运行脚本)
    	--README.md
    	--requirements.txt

### 包依赖

---

- python3.*
- pandas
- numpy
- matplotlib

### 包依赖安装

---

在项目根目录使用如下命令进行python环境配置

```
pip3 install -r requirements.txt
```

### 运行

---

在train路径下通过使用脚本来运行程序，运行脚本前需要先赋予执行权限，即输入以下命令：

```
cd train
chmod +x run.sh
```

执行脚本

```
./run.sh
```

### 说明

---

**代码结构**

```
功能组件：
	baseic_model.py：基本的网络结构
	...
主流程相关：
	data_process.py:数据预处理
	train.py:模型训练
	...
辅助工具
	...
执行脚本
	run.sh：执行整个模型
```

**数据说明**

数据放置于data_batching目录下

```
--data_batching
	--raw_data
		-0.csv
		...
	--clean_data
```

**超参数说明**

可在hyperparameters_manager.py中修改

```
train_ratio = 0.95 --训练集中训练数据占比

num_epochs = 100 --训练批次

batch_size = 128 --每批训练集大小

lr_decay_rate = 0.95 --学习率衰减率

initial_lr = 0.001 --初始学习率
```

