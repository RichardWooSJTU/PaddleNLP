# LLaMA DyBatch

## 分割权重
```
bash split_weight.sh {input_model_dir} {output_model_dir}
```

## 动态图推理
```
bash run.sh {input_model_dir}
```

## 动转静导出
```
bash export.sh.sh {input_model_dir} {inference_model_dir}
```

## 静态图推理
```
bash inference.sh {inference_model_dir}
```