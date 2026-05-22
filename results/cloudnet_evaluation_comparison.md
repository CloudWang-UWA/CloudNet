# CloudNet Evaluation Comparison

Relative delta uses `(CloudNet - baseline) / CloudNet`.
Missing values are shown as `/`.

## Main Comparison

| Method                  | CLIP-Text | CLIP-Image | CLIP-AES | mIoU    | Note                                                 |
| ----------------------- | --------- | ---------- | -------- | ------- | ---------------------------------------------------- |
| Original-IP, controlled | 24.9538   | 0.8276     | 6.0293   | 29.3300 | 100 steps, scale 0.8; preferred Original-IP baseline |
| Original-IP, historical | 24.7687   | 0.7916     | 6.0434   | 24.9000 | Jan reference only; 70 steps                         |
| FreestyleNet            | 26.4447   | /          | 5.6835   | 27.5300 | No CLIP-I result available                           |
| CloudNet ckpt500        | 24.8881   | 0.8352     | 6.1166   | 30.2600 | Original token5 checkpoint                           |
| CloudNet ckpt6500       | 24.8286   | 0.8359     | 6.1487   | 28.9800 | Current main model                                   |

## Vs FreestyleNet

| Style     | CLIP-Text |              |           | CLIP-Image |              |           | CLIP-AES |              |           | mIoU     |              |           |
| --------- | --------- | ------------ | --------- | ---------- | ------------ | --------- | -------- | ------------ | --------- | -------- | ------------ | --------- |
|           | CloudNet  | FreestyleNet | Delta (%) | CloudNet   | FreestyleNet | Delta (%) | CloudNet | FreestyleNet | Delta (%) | CloudNet | FreestyleNet | Delta (%) |
| Cozy      | 25.5248   | 26.5076      | -3.85%    | 0.8755     | /            | /         | 6.1824   | 5.5121       | 10.84%    | 33.4600  | 33.6300      | -0.51%    |
| Messy     | 24.1751   | 27.2103      | -12.56%   | 0.7359     | /            | /         | 5.4778   | 5.5604       | -1.51%    | 29.8400  | 27.1900      | 8.88%     |
| Classic   | 25.3165   | 24.9229      | 1.55%     | 0.8589     | /            | /         | 6.2464   | 5.7801       | 7.46%     | 34.5600  | 29.5000      | 14.64%    |
| Luxurious | 25.7050   | 25.4575      | 0.96%     | 0.8957     | /            | /         | 6.0467   | 5.5378       | 8.42%     | 32.3600  | 32.6100      | -0.77%    |
| Van_gogh  | 23.4217   | 28.1252      | -20.08%   | 0.8136     | /            | /         | 6.7902   | 6.0268       | 11.24%    | 26.3900  | 25.5200      | 3.30%     |
| Overall   | 24.8286   | 26.4447      | -6.51%    | 0.8359     | /            | /         | 6.1487   | 5.6835       | 7.57%     | 28.9800  | 27.5300      | 5.00%     |

## Style-IP vs Original-IP Controlled

| Style     | CLIP-Text |             |           | CLIP-Image |             |           | CLIP-AES |             |           | mIoU     |             |           |
| --------- | --------- | ----------- | --------- | ---------- | ----------- | --------- | -------- | ----------- | --------- | -------- | ----------- | --------- |
|           | Style-IP  | Original-IP | Delta (%) | Style-IP   | Original-IP | Delta (%) | Style-IP | Original-IP | Delta (%) | Style-IP | Original-IP | Delta (%) |
| Cozy      | 25.5248   | 25.6388     | -0.45%    | 0.8755     | 0.8746      | 0.10%     | 6.1824   | 6.0221      | 2.59%     | 33.4600  | 33.3800     | 0.24%     |
| Messy     | 24.1751   | 24.2596     | -0.35%    | 0.7359     | 0.7245      | 1.55%     | 5.4778   | 5.5383      | -1.11%    | 29.8400  | 30.8200     | -3.28%    |
| Classic   | 25.3165   | 25.4887     | -0.68%    | 0.8589     | 0.8527      | 0.72%     | 6.2464   | 6.0849      | 2.59%     | 34.5600  | 34.9900     | -1.24%    |
| Luxurious | 25.7050   | 25.5760     | 0.50%     | 0.8957     | 0.8874      | 0.93%     | 6.0467   | 5.9184      | 2.12%     | 32.3600  | 30.7800     | 4.88%     |
| Van_gogh  | 23.4217   | 23.8061     | -1.64%    | 0.8136     | 0.7989      | 1.81%     | 6.7902   | 6.5829      | 3.05%     | 26.3900  | 26.1200     | 1.02%     |
| Overall   | 24.8286   | 24.9538     | -0.50%    | 0.8359     | 0.8276      | 0.99%     | 6.1487   | 6.0293      | 1.94%     | 28.9800  | 29.3300     | -1.21%    |

## Style-IP vs Original-IP Historical

| Style     | CLIP-Text |                        |           | CLIP-Image |                        |           | CLIP-AES |                        |           | mIoU     |                        |           |
| --------- | --------- | ---------------------- | --------- | ---------- | ---------------------- | --------- | -------- | ---------------------- | --------- | -------- | ---------------------- | --------- |
|           | Style-IP  | Historical Original-IP | Delta (%) | Style-IP   | Historical Original-IP | Delta (%) | Style-IP | Historical Original-IP | Delta (%) | Style-IP | Historical Original-IP | Delta (%) |
| Cozy      | 25.5248   | 25.3582                | 0.65%     | 0.8755     | 0.8472                 | 3.23%     | 6.1824   | 5.9187                 | 4.26%     | 33.4600  | 28.8000                | 13.93%    |
| Messy     | 24.1751   | 23.5350                | 2.65%     | 0.7359     | 0.7141                 | 2.96%     | 5.4778   | 5.7386                 | -4.76%    | 29.8400  | 27.5700                | 7.61%     |
| Classic   | 25.3165   | 25.1425                | 0.69%     | 0.8589     | 0.7676                 | 10.63%    | 6.2464   | 6.3391                 | -1.48%    | 34.5600  | 27.5500                | 20.28%    |
| Luxurious | 25.7050   | 25.7663                | -0.24%    | 0.8957     | 0.8779                 | 1.99%     | 6.0467   | 5.9350                 | 1.85%     | 32.3600  | 29.2600                | 9.58%     |
| Van_gogh  | 23.4217   | 24.0416                | -2.65%    | 0.8136     | 0.7510                 | 7.70%     | 6.7902   | 6.2854                 | 7.43%     | 26.3900  | 22.4000                | 15.12%    |
| Overall   | 24.8286   | 24.7687                | 0.24%     | 0.8359     | 0.7916                 | 5.31%     | 6.1487   | 6.0434                 | 1.71%     | 28.9800  | 24.9000                | 14.08%    |

## Checkpoint Selection

| Checkpoint | CLIP-Text | CLIP-Image | CLIP-AES | mIoU    | Comment                     |
| ---------- | --------- | ---------- | -------- | ------- | --------------------------- |
| ckpt500    | 24.8881   | 0.8352     | 6.1166   | 30.2600 | Original token5             |
| ckpt1500   | 24.8040   | 0.8369     | 6.1312   | /       | Style checkpoint ablation   |
| ckpt3000   | 24.8796   | 0.8359     | 6.1236   | /       | Style checkpoint ablation   |
| ckpt5000   | 24.8831   | 0.8356     | 6.0934   | /       | Style checkpoint ablation   |
| ckpt6500   | 24.8286   | 0.8359     | 6.1487   | 28.9800 | Best CLIP-AES; current main |

## Style Scale Sensitivity

| Style token scale | CLIP-Text | CLIP-Image | CLIP-AES | mIoU    | Comment                       |
| ----------------- | --------- | ---------- | -------- | ------- | ----------------------------- |
| 0.5               | 24.8350   | 0.8350     | 6.1509   | 29.0600 | Inference-only sensitivity    |
| 0.8               | 24.8116   | 0.8361     | 6.1582   | 29.2600 | Inference-only sensitivity    |
| 1.0               | 24.8286   | 0.8359     | 6.1487   | 28.9800 | Default/main ckpt6500 setting |
| 1.2               | 24.8151   | 0.8362     | 6.1510   | 29.9100 | Inference-only sensitivity    |
| 1.5               | 24.8758   | 0.8357     | 6.1434   | 29.0000 | Inference-only sensitivity    |

## mIoU Stylewise

| Method                  | Overall | Cozy    | Messy   | Classic | Luxurious | Van_gogh |
| ----------------------- | ------- | ------- | ------- | ------- | --------- | -------- |
| Original-IP, controlled | 29.3300 | 33.3800 | 30.8200 | 34.9900 | 30.7800   | 26.1200  |
| Original-IP, historical | 24.9000 | 28.8000 | 27.5700 | 27.5500 | 29.2600   | 22.4000  |
| FreestyleNet            | 27.5300 | 33.6300 | 27.1900 | 29.5000 | 32.6100   | 25.5200  |
| CloudNet ckpt500        | 30.2600 | 33.9100 | 30.4600 | 34.6000 | 33.1200   | 27.4300  |
| CloudNet ckpt6500       | 28.9800 | 33.4600 | 29.8400 | 34.5600 | 32.3600   | 26.3900  |

## Notes

| Item                         | Value                                                                      |
| ---------------------------- | -------------------------------------------------------------------------- |
| CloudNet main model          | 2026-05-02_style-ipadapter-token5_ckpt6500_controlnet97_100steps_scale0.8  |
| Delta convention             | (CloudNet - baseline) / CloudNet                                           |
| FreestyleNet CLIP-Image      | Not available in current evaluation outputs                                |
| Normal IP-Adapter baseline   | Use the controlled 2026-05-03 100-step/scale-0.8 run for formal comparison |
| Historical normal IP-Adapter | Keep the Jan 70-step run as reference only                                 |
