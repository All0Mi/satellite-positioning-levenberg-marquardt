function [dop, elev_deg, azim_deg] = dop_metrics(Sat, p, lat0_deg, lon0_deg)
    los = Sat - p(:)';
    los = los ./ vecnorm(los, 2, 2);        % unit vectors receiver -> satellite
    E   = ecef2enu(los, lat0_deg, lon0_deg);

    elev_deg = rad2deg(asin(E(:,3)));
    azim_deg = mod(rad2deg(atan2(E(:,1), E(:,2))), 360);

    % Position-only geometry (3 unknowns: x, y, z)
    Q3 = inv(E' * E);
    dop.HDOP = sqrt(Q3(1,1) + Q3(2,2));
    dop.VDOP = sqrt(Q3(3,3));
    dop.PDOP = sqrt(trace(Q3));

    % Classic GPS geometry including a receiver clock bias
    A4 = [-E, ones(size(E,1),1)];
    Q4 = inv(A4' * A4);
    dop.GDOP  = sqrt(trace(Q4));
    dop.TDOP  = sqrt(Q4(4,4));
end
