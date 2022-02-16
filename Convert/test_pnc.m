var_res = [];
mean_res = [];
for i = 1:812
    vec = rest_fmri_power264(i).img_time_serie(:,:);
    mean_res(i) = mean(vec(:));
    var_res(i) = mean(var(vec'));
end