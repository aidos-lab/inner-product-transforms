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
| Encoder | 1.003551 | 4.434669 | 6.006771 | 9.393849 | 9.626629 | 17.237462 | 48.546912 | 57.837475 |
| Encoder Chamfer | 0.978317 | 9.759554 | 6.043088 | 13.437557 | 9.774287 | 24.118090 | NaN | NaN |
| Rendered | 0.316157 | 0.116938 | 0.889835 | 0.273740 | 0.683799 | 0.221733 | NaN | NaN |

</div>

# Model outputs

## MNIST

![](readme_files/figure-commonmark/cell-6-output-1.png)

## CNN Encoder Chair

![](readme_files/figure-commonmark/cell-7-output-1.png)

## CNN Encoder Char

![](readme_files/figure-commonmark/cell-8-output-1.png)

## CNN Encoder Airplane

![](readme_files/figure-commonmark/cell-9-output-1.png)
