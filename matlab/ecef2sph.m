function [lat_deg, lon_deg, h] = ecef2sph(P, R)
    if isvector(P) && numel(P) == 3
        P = reshape(P, 1, 3);
    end
    X = P(:,1); Y = P(:,2); Z = P(:,3);
    r = sqrt(X.^2 + Y.^2 + Z.^2);

    lat_deg = rad2deg(asin(Z ./ r));
    lon_deg = rad2deg(atan2(Y, X));
    h = r - R;
end
