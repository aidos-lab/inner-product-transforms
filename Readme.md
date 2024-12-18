# Reconstruction of MNIST and ShapeNetCore


The encoder is the `encoder_sparse` and the rendered model is the point
cloud optimization rendering (a non-parametric method). The latter
clearly outperforms all models.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead tr th {
        text-align: left;
    }
&#10;    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>

| dataset | Airplane |  | Car |  | Chair |  | Mnist |  |
|----|----|----|----|----|----|----|----|----|
|  | MMD-CD | MMD-EMD | MMD-CD | MMD-EMD | MMD-CD | MMD-EMD | MMD-CD | MMD-EMD |
|  | mean | mean | mean | mean | mean | mean | mean | mean |
| model |  |  |  |  |  |  |  |  |
| Ect-128 | 3.382390 | 0.466779 | 6.221755 | 0.845426 | 5.716591 | 0.821442 | NaN | NaN |
| Ect-256 | 0.316157 | 0.116938 | 0.889835 | 0.273740 | 0.683799 | 0.221733 | NaN | NaN |
| Ect-64 | 6.810036 | 0.965796 | 9.268112 | 1.347526 | 21.936062 | 2.929956 | NaN | NaN |
| Encoder | 1.012534 | 1.499121 | 6.007284 | 4.382427 | 9.525050 | 8.446839 | 41.341307 | 11.577831 |
| Encoder Chamfer | 1.001583 | 8.894593 | 5.971047 | 14.959719 | 10.439118 | 32.430265 | NaN | NaN |
| Encoder Downsample | 2.365263 | 26.006261 | 11.392715 | 41.941542 | 16.051830 | 75.206883 | 159.988590 | 221.338034 |
| Encoder Ect | 2.413157 | 1.088348 | 7.749730 | 2.471831 | 13.056453 | 4.289896 | NaN | NaN |

</div>

# Model outputs

## MNIST

![](Readme_files/figure-commonmark/cell-6-output-1.png)

![](Readme_files/figure-commonmark/cell-7-output-1.png)

![](Readme_files/figure-commonmark/cell-9-output-1.png)

## CNN Encoder Chair (ECT + Chamfer distance loss)

![](Readme_files/figure-commonmark/cell-10-output-1.png)

## CNN Encoder Chair (ECT + Chamfer loss)

![](Readme_files/figure-commonmark/cell-11-output-1.png)

# Encoder with only ECT loss

![](Readme_files/figure-commonmark/cell-12-output-1.png)

![](Readme_files/figure-commonmark/cell-13-output-1.png)

![](Readme_files/figure-commonmark/cell-14-output-1.png)

## CNN Encoder Airplane

## ECT-64

![](Readme_files/figure-commonmark/cell-16-output-1.png)

![](Readme_files/figure-commonmark/cell-16-output-2.png)

![](Readme_files/figure-commonmark/cell-16-output-3.png)

# Downsample Airplane

![](Readme_files/figure-commonmark/cell-17-output-1.png)

![](Readme_files/figure-commonmark/cell-18-output-1.png)

![](Readme_files/figure-commonmark/cell-19-output-1.png)
