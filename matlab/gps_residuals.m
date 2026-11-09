function F = gps_residuals(p, Sat, d)
    diffs = Sat - p(:)';
    dist  = sqrt(sum(diffs.^2, 2));
    F = dist - d(:);
end
