select brand,
        count(sku_type) as count_sku
from sku_dict_another_one
where brand != 0
group by brand
order by count_sku DESC
limit 10
