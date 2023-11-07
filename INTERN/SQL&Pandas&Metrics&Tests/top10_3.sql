select vendor,
       count(distinct brand) as brand
from sku_dict_another_one
where brand is not null
group by vendor
order by brand DESC
limit 10
