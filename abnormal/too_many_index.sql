create index toomany1 on bmsql_config(cfg_value) local;
create index toomany2 on bmsql_config(cfg_name) local;
create index toomany3 on bmsql_customer(c_w_id) local;
create index toomany4 on bmsql_customer(c_d_id) local;
create index toomany5 on bmsql_customer(c_id) local;
create index toomany6 on bmsql_district(d_w_id) local;
create index toomany7 on bmsql_district(d_id) local;
create index toomany8 on bmsql_history(h_c_id) local;
create index toomany9 on bmsql_history(h_c_d_id) local;
create index toomany10 on bmsql_item(i_id) local;
create index toomany11 on bmsql_item(i_name) local;
create index toomany12 on bmsql_new_order(no_w_id) local;
create index toomany13 on bmsql_new_order(no_d_id) local;
create index toomany14 on bmsql_oorder(o_w_id) local;
create index toomany15 on bmsql_oorder(o_d_id) local;
create index toomany16 on bmsql_order_line(ol_w_id) local;
create index toomany17 on bmsql_order_line(ol_d_id) local;
create index toomany18 on bmsql_stock(s_w_id) local;
create index toomany19 on bmsql_stock(s_i_id) local;
create index toomany20 on bmsql_warehouse(w_id) local;