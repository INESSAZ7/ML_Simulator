select vendor,
       count(sku_type) as sku
from sku_dict_another_one
where vendor is not null
group by vendor
order by sku DESC
limit 10
