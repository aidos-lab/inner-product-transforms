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

| dataset | Airplane |  | Car |  | Chair |  | MNIST |  |
|----|----|----|----|----|----|----|----|----|
|  | MMD-CD | MMD-EMD | MMD-CD | MMD-EMD | MMD-CD | MMD-EMD | MMD-CD | MMD-EMD |
|  | mean | mean | mean | mean | mean | mean | mean | mean |
| model |  |  |  |  |  |  |  |  |
| Encoder | 1.121926 | 4.798106 | 6.146839 | 10.299724 | 9.820879 | 25.073627 | 48.546912 | 57.837475 |
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
