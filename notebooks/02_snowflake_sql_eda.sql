-- =====================================================
-- QUERY 1: Rolling 3-Month Average
-- =====================================================
WITH monthly_stats AS (
    SELECT
        ARRIVAL_DATE_YEAR,
        ARRIVAL_DATE_MONTH,

        CASE ARRIVAL_DATE_MONTH
            WHEN 'January' THEN 1
            WHEN 'February' THEN 2
            WHEN 'March' THEN 3
            WHEN 'April' THEN 4
            WHEN 'May' THEN 5
            WHEN 'June' THEN 6
            WHEN 'July' THEN 7
            WHEN 'August' THEN 8
            WHEN 'September' THEN 9
            WHEN 'October' THEN 10
            WHEN 'November' THEN 11
            WHEN 'December' THEN 12
        END AS MONTH_NUMBER,

        COUNT(*) AS BOOKINGS,
        SUM(IS_CANCELED) AS CANCELLATIONS

    FROM TRAVELIQ_DB.ANALYTICS.HOTEL_BOOKINGS_CLEAN

    GROUP BY
        ARRIVAL_DATE_YEAR,
        ARRIVAL_DATE_MONTH
)

SELECT
    ARRIVAL_DATE_YEAR,
    ARRIVAL_DATE_MONTH,
    BOOKINGS,
    CANCELLATIONS,

    ROUND(
        CANCELLATIONS * 100.0 / NULLIF(BOOKINGS, 0),
        2
    ) AS CANCELLATION_RATE_PCT,

    ROUND(
        AVG(BOOKINGS) OVER (
            ORDER BY ARRIVAL_DATE_YEAR, MONTH_NUMBER
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ),
        2
    ) AS ROLLING_3_MONTH_AVG

FROM monthly_stats

ORDER BY
    ARRIVAL_DATE_YEAR,
    MONTH_NUMBER;


-- =====================================================
-- QUERY 2: Hotel-wise Revenue at Risk
-- =====================================================
SELECT
    HOTEL,

    COUNT(*) AS TOTAL_BOOKINGS,

    SUM(IS_CANCELED) AS CANCELLED_BOOKINGS,

    ROUND(
        SUM(
            CASE
                WHEN IS_CANCELED = 1 THEN
                    ADR * (
                        STAYS_IN_WEEKEND_NIGHTS
                        + STAYS_IN_WEEK_NIGHTS
                    )
                ELSE 0
            END
        ),
        2
    ) AS REVENUE_AT_RISK,

    ROUND(
        SUM(
            ADR * (
                STAYS_IN_WEEKEND_NIGHTS
                + STAYS_IN_WEEK_NIGHTS
            )
        ),
        2
    ) AS TOTAL_POTENTIAL_ROOM_REVENUE,

    ROUND(
        SUM(
            CASE
                WHEN IS_CANCELED = 1 THEN
                    ADR * (
                        STAYS_IN_WEEKEND_NIGHTS
                        + STAYS_IN_WEEK_NIGHTS
                    )
                ELSE 0
            END
        ) * 100.0
        /
        NULLIF(
            SUM(
                ADR * (
                    STAYS_IN_WEEKEND_NIGHTS
                    + STAYS_IN_WEEK_NIGHTS
                )
            ),
            0
        ),
        2
    ) AS PCT_REVENUE_AT_RISK

FROM TRAVELIQ_DB.ANALYTICS.HOTEL_BOOKINGS_CLEAN

WHERE ADR IS NOT NULL

GROUP BY HOTEL

ORDER BY REVENUE_AT_RISK DESC;


-- =====================================================
-- QUERY 3: Country-wise Cancellation Risk Ranking
-- =====================================================
WITH country_stats AS (
    SELECT
        COUNTRY,
        COUNT(*) AS TOTAL_BOOKINGS,
        ROUND(AVG(IS_CANCELED) * 100, 2) AS CANCELLATION_RATE_PCT
    FROM TRAVELIQ_DB.ANALYTICS.HOTEL_BOOKINGS_CLEAN
    WHERE COUNTRY != 'Unknown'
    GROUP BY COUNTRY
    HAVING COUNT(*) >= 100
)

SELECT
    COUNTRY,
    TOTAL_BOOKINGS,
    CANCELLATION_RATE_PCT,

    RANK() OVER (
        ORDER BY CANCELLATION_RATE_PCT DESC
    ) AS RISK_RANK

FROM country_stats

ORDER BY RISK_RANK

LIMIT 10;


-- =====================================================
-- QUERY 4: Running Total of Cancellations
-- =====================================================
WITH daily_stats AS (
    SELECT
        RESERVATION_STATUS_DATE,
        COUNT(*) AS BOOKINGS_ON_DATE,
        SUM(IS_CANCELED) AS CANCELLATIONS_ON_DATE
    FROM TRAVELIQ_DB.ANALYTICS.HOTEL_BOOKINGS_CLEAN
    GROUP BY RESERVATION_STATUS_DATE
)

SELECT
    RESERVATION_STATUS_DATE,
    BOOKINGS_ON_DATE,
    CANCELLATIONS_ON_DATE,

    SUM(CANCELLATIONS_ON_DATE) OVER (
        ORDER BY RESERVATION_STATUS_DATE
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS RUNNING_TOTAL_CANCELLATIONS

FROM daily_stats

ORDER BY RESERVATION_STATUS_DATE

LIMIT 50;


-- =====================================================
-- QUERY 5: Lead-Time Risk Buckets
-- =====================================================
WITH risk_bucket AS (
    SELECT
        LEAD_TIME,

        IS_CANCELED,

        CASE
            WHEN LEAD_TIME < 30 THEN 'Low (0-29 days)'
            WHEN LEAD_TIME < 90 THEN 'Medium (30-89 days)'
            WHEN LEAD_TIME < 180 THEN 'High (90-179 days)'
            ELSE 'Very High (180+ days)'
        END AS LEAD_TIME_RISK

    FROM TRAVELIQ_DB.ANALYTICS.HOTEL_BOOKINGS_CLEAN
)

SELECT
    LEAD_TIME_RISK,

    COUNT(*) AS TOTAL_BOOKINGS,

    SUM(IS_CANCELED) AS CANCELLED_BOOKINGS,

    ROUND(
        AVG(IS_CANCELED) * 100,
        2
    ) AS CANCELLATION_RATE_PCT,

    ROUND(
        COUNT(*) * 100.0
        / SUM(COUNT(*)) OVER (),
        2
    ) AS PCT_OF_TOTAL_BOOKINGS

FROM risk_bucket

GROUP BY LEAD_TIME_RISK

ORDER BY CANCELLATION_RATE_PCT DESC;


-- =====================================================
-- QUERY 6: Hotel-wise Cancellation Contribution
-- =====================================================
WITH hotel_cancellations AS (
    SELECT
        HOTEL,
        COUNT(*) AS TOTAL_BOOKINGS,
        SUM(IS_CANCELED) AS CANCELLED_BOOKINGS
    FROM TRAVELIQ_DB.ANALYTICS.HOTEL_BOOKINGS_CLEAN
    GROUP BY HOTEL
)

SELECT
    HOTEL,
    TOTAL_BOOKINGS,
    CANCELLED_BOOKINGS,

    ROUND(
        CANCELLED_BOOKINGS * 100.0
        / NULLIF(TOTAL_BOOKINGS, 0),
        2
    ) AS CANCELLATION_RATE_PCT,

    ROUND(
        CANCELLED_BOOKINGS * 100.0
        / NULLIF(
            SUM(CANCELLED_BOOKINGS) OVER (),
            0
        ),
        2
    ) AS PCT_CONTRIBUTION_TO_TOTAL_CANCELLATIONS

FROM hotel_cancellations

ORDER BY PCT_CONTRIBUTION_TO_TOTAL_CANCELLATIONS DESC;


-- =====================================================
-- QUERY 7: Executive Summary
-- =====================================================
SELECT
    COUNT(*) AS TOTAL_BOOKINGS,

    SUM(IS_CANCELED) AS CANCELLED_BOOKINGS,

    ROUND(
        AVG(IS_CANCELED) * 100,
        2
    ) AS CANCELLATION_RATE_PCT,

    ROUND(
        SUM(
            ADR * (
                STAYS_IN_WEEKEND_NIGHTS
                + STAYS_IN_WEEK_NIGHTS
            )
        ),
        2
    ) AS TOTAL_POTENTIAL_ROOM_REVENUE,

    ROUND(
        SUM(
            CASE
                WHEN IS_CANCELED = 1 THEN
                    ADR * (
                        STAYS_IN_WEEKEND_NIGHTS
                        + STAYS_IN_WEEK_NIGHTS
                    )
                ELSE 0
            END
        ),
        2
    ) AS REVENUE_AT_RISK,

    ROUND(
        SUM(
            CASE
                WHEN IS_CANCELED = 1 THEN
                    ADR * (
                        STAYS_IN_WEEKEND_NIGHTS
                        + STAYS_IN_WEEK_NIGHTS
                    )
                ELSE 0
            END
        ) * 100.0
        /
        NULLIF(
            SUM(
                ADR * (
                    STAYS_IN_WEEKEND_NIGHTS
                    + STAYS_IN_WEEK_NIGHTS
                )
            ),
            0
        ),
        2
    ) AS PCT_REVENUE_AT_RISK

FROM TRAVELIQ_DB.ANALYTICS.HOTEL_BOOKINGS_CLEAN

WHERE ADR IS NOT NULL;


WITH adr_buckets AS (
    SELECT
        *,
        CASE
            WHEN ADR < 50 THEN 'Low (<50)'
            WHEN ADR < 100 THEN 'Medium (50-99)'
            WHEN ADR < 150 THEN 'High (100-149)'
            ELSE 'Very High (150+)'
        END AS ADR_BUCKET
    FROM TRAVELIQ_DB.ANALYTICS.HOTEL_BOOKINGS_CLEAN
    WHERE ADR IS NOT NULL
)


-- =====================================================
-- QUERY 8: ADR Bucket vs Cancellation Risk
-- =====================================================
SELECT
    ADR_BUCKET,

    COUNT(*) AS TOTAL_BOOKINGS,

    SUM(IS_CANCELED) AS CANCELLED_BOOKINGS,

    ROUND(
        AVG(IS_CANCELED) * 100,
        2
    ) AS CANCELLATION_RATE_PCT,

    ROUND(
        COUNT(*) * 100.0
        / SUM(COUNT(*)) OVER (),
        2
    ) AS PCT_OF_TOTAL_BOOKINGS

FROM adr_buckets

GROUP BY ADR_BUCKET

ORDER BY CANCELLATION_RATE_PCT DESC;


-- =====================================================
-- QUERY 9: Month-over-Month Booking Change
-- =====================================================
WITH monthly_bookings AS (
    SELECT
        ARRIVAL_DATE_YEAR,
        ARRIVAL_DATE_MONTH,
        CASE ARRIVAL_DATE_MONTH
            WHEN 'January' THEN 1
            WHEN 'February' THEN 2
            WHEN 'March' THEN 3
            WHEN 'April' THEN 4
            WHEN 'May' THEN 5
            WHEN 'June' THEN 6
            WHEN 'July' THEN 7
            WHEN 'August' THEN 8
            WHEN 'September' THEN 9
            WHEN 'October' THEN 10
            WHEN 'November' THEN 11
            WHEN 'December' THEN 12
        END AS MONTH_NUMBER,
        COUNT(*) AS BOOKINGS
    FROM TRAVELIQ_DB.ANALYTICS.HOTEL_BOOKINGS_CLEAN
    GROUP BY
        ARRIVAL_DATE_YEAR,
        ARRIVAL_DATE_MONTH
)

SELECT
    ARRIVAL_DATE_YEAR,
    ARRIVAL_DATE_MONTH,
    BOOKINGS,

    LAG(BOOKINGS) OVER (
        ORDER BY ARRIVAL_DATE_YEAR, MONTH_NUMBER
    ) AS PREV_MONTH_BOOKINGS,

    BOOKINGS
    - LAG(BOOKINGS) OVER (
        ORDER BY ARRIVAL_DATE_YEAR, MONTH_NUMBER
    ) AS MONTH_OVER_MONTH_CHANGE

FROM monthly_bookings

ORDER BY
    ARRIVAL_DATE_YEAR,
    MONTH_NUMBER;
