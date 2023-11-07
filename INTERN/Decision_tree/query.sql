select  age,
        income,
        dependents,
        has_property,
        has_car,
        credit_score,
        job_tenure,
        has_education,
        loan_amount,
        date_diff(day,loan_start,loan_deadline) as loan_period,
        if (date_diff(day, loan_deadline, loan_payed) > 0, date_diff(day,loan_deadline,loan_payed), 0) as delay_days
from default.loan_delay_days