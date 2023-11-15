-- Query 1

SELECT 
    gt.*,
    lkup_pu."Original_Value" AS pu_location_name,
    lkup_do."Original_Value" AS do_location_name
FROM 
    green_taxi_4_2015 AS gt 
INNER JOIN 
    lookup_green_taxi_4_2015 AS lkup_pu ON gt.pu_location = lkup_pu."Imputed_Value"::bigint
INNER JOIN 
    lookup_green_taxi_4_2015 AS lkup_do ON gt.do_location = lkup_do."Imputed_Value"::bigint
WHERE 
    lkup_pu."Column" = 'pu_location' AND lkup_do."Column" = 'do_location'
ORDER BY 
    gt.trip_distance DESC
LIMIT 
    20;





-- Query 2

select payment_type, avg(fare_amount)
from green_taxi_4_2015
group by payment_type;  


-- Query 3

SELECT
    SPLIT_PART(lkup_pu."Original_Value", ',', 1) AS pu_city,
    AVG(tip_amount) AS average_tip
FROM
    green_taxi_4_2015 AS gt
INNER JOIN
    lookup_green_taxi_4_2015 AS lkup_pu ON gt.pu_location = lkup_pu."Imputed_Value"::bigint
WHERE
    lkup_pu."Column" = 'pu_location'
GROUP BY
    pu_city
ORDER BY
    average_tip DESC
LIMIT
    1



-- Query 4

SELECT
    SPLIT_PART(lkup_pu."Original_Value", ',', 1) AS pu_city,
    AVG(tip_amount) AS average_tip
FROM
    green_taxi_4_2015 AS gt
INNER JOIN
    lookup_green_taxi_4_2015 AS lkup_pu ON gt.pu_location = lkup_pu."Imputed_Value"::bigint
WHERE
    lkup_pu."Column" = 'pu_location'
GROUP BY
    pu_city
ORDER BY
    average_tip ASC
LIMIT
    1

-- Query 5

SELECT 
    lkup_do."Original_Value" AS destination, count(*) AS weekend_trips_count
FROM 
    green_taxi_4_2015 AS gt 
INNER JOIN
    lookup_green_taxi_4_2015 AS lkup_do ON gt.do_location = lkup_do."Imputed_Value"::bigint
WHERE 
    lkup_do."Column" = 'do_location' and weekend_trip = 1
GROUP BY destination
ORDER BY weekend_trips_count DESC
LIMIT 1;


-- Query 6

SELECT
  trip_type,
  AVG(trip_distance) AS avg_distance
FROM
  green_taxi_4_2015
GROUP BY
  trip_type
ORDER BY
  avg_distance DESC;
  
  


-- Query 7

SELECT avg(fare_amount) AS avg_fare_amount
FROM (
    SELECT
        date_part('hour', lpep_pickup_datetime::TIMESTAMP) AS pu_hour, fare_amount, lpep_pickup_datetime
    FROM
        green_taxi_4_2015
) AS pickup
where pu_hour >=16 and pu_hour <18
