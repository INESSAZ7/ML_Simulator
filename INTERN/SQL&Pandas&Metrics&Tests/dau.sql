select toDate(timestamp) as day,
        uniqExact(user_id) as dau 
from default.churn_submits
group by day
