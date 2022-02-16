%% write_wrat_csv.m

% Write WRAT scores from cnb_data.mat to CSV files
% keyed on subject ID

% data is found cnb_data struct
% If first column starts with ID excel will mistake .csv for SYLK file

fname = 'wrat2.csv';
fid = fopen(fname, 'w');
fprintf(fid, 'PNCID,Valid,Raw,Std\n');

for i=1:length(cnp_data)
    id = cnp_data(i).meta.id;
    valid = cnp_data(i).res_wrat.wrat_valid;
    raw = cnp_data(i).res_wrat.wrat_cr_raw;
    std = cnp_data(i).res_wrat.wrat_cr_std;
    
    fprintf(fid, "%s,%s,%d,%d\n",id,valid,raw,std);
end

fclose(fid);