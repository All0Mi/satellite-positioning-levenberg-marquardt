function J = gps_jacobian(p, Sat)
    diffs = Sat - p(:)';
    dist  = sqrt(sum(diffs.^2, 2));
    J = -diffs ./ dist;
end
