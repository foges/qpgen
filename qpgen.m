classdef qpgen < handle
  properties (Hidden = true, SetAccess = private)
    cpp_handle;
  end
  methods
    % Constructor
    function this = qpgen(data)
      this.cpp_handle = qpgen_init(data);
    end
    % Destructor
    function delete(this)
      qpgen_clear(this.cpp_handle);
    end
    % Example method
    function output = run(this, data, max_iter)
      output = qpgen_run(this.cpp_handle, data, max_iter);
    end
  end
end