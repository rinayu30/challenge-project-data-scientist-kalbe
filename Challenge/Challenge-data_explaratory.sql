SELECT "Marital Status" ,AVG(Age) AS rata_rata_umur FROM customer group BY "Marital Status";
select * from kalbe.customer
select * from kalbe.product
select * from kalbe.store
select * from kalbe."transaction"

/**
QUERY UNTUK MENENTUKAN RATA-RATA UMUR PELANGGAN BERDASARKAN MARITAL STATUS
**/
SELECT 
    CASE 
        WHEN "Marital Status" = '' THEN 'Unknown'
        ELSE "Marital Status"
    END AS marital_status,
    FLOOR(AVG(age)) AS avg_age
FROM kalbe.customer
GROUP BY marital_status;
/**
QUERY UNTUK MENENTUKAN RATA-RATA UMUR PELANGGAN BERDASARKAN GENDER
**/
SELECT 
    CASE 
        WHEN gender = 0 THEN 'Female'
        WHEN gender = 1 THEN 'Male'
    END AS customer_gender,
    FLOOR(AVG(age)) AS avg_age
FROM kalbe.customer
GROUP BY gender;
/**
QUERY UNTUK MENENTUKAN NAMA STORE DENGAN TOTAL QTY TERBANYAK
**/
SELECT s.storename, sum(t.qty) AS total_qty
FROM kalbe.store s
JOIN kalbe.transaction t  ON t.storeid  = s.storeid
GROUP BY s.storename 
ORDER BY total_qty desc
LIMIT 1;
/**
QUERY UNTUK MENENTUKAN NAMA PRODUK TERLARIS DENGAN TOTAL AMT TERBANYAK
**/
SELECT p."Product Name", sum(t.totalamount) AS "Total Amount"
FROM kalbe.product p
JOIN kalbe.transaction t ON t.productid = p.productid
GROUP BY p."Product Name" 
ORDER BY "Total Amount" desc
LIMIT 1;