function [ city, field, forest, grass, street ] = very_simple_data_loader( size, ch, num )
    d = floor(ch/6);
    s = floor(size/16);
    % create very simple data 
    city = complex(rand(size, size, ch, num), rand(size, size, ch, num)) * 1e-3;
    field = complex(rand(size, size, ch, num), rand(size, size, ch, num)) * 1e-3;
    forest = complex(rand(size, size, ch, num), rand(size, size, ch, num)) * 1e-3;
    grass = complex(rand(size, size, ch, num), rand(size, size, ch, num)) * 1e-3;
    street = complex(rand(size, size, ch, num), rand(size, size, ch, num)) * 1e-3;

    city(1:s, 1:s, :, :) = city(1:s, 1:s, :, :)*1e3;
    field(s+1:2*s, s+1:2*s, :, :) = field(s+1:2*s, s+1:2*s, :, :)*1e3;
    forest(2*s+1:3*s, 2*s+1:3*s, :, :) = forest(2*s+1:3*s, 2*s+1:3*s, :, :)*1e3;
    grass(3*s+1:4*s, 3*s+1:4*s, :, :) = grass(3*s+1:4*s, 3*s+1:4*s, :, :)*1e3;
    street(4*s+1:5*s, 4*s+1:5*s, :, :) = street(4*s+1:5*s, 4*s+1:5*s, :, :)*1e3;
end

