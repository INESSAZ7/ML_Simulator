select toStartOfMonth(toDate(buy_date)) as month,
        avg(check_amount) as avg_check,
        quantileExactExclusive(0.5)(check_amount) as median_check
from default.view_checks
group by month
