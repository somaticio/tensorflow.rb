$LOAD_PATH.unshift File.expand_path('./../lib', __FILE__)
require 'simplecov'
require 'simplecov-json'
require 'simplecov-rcov'
require 'pry'

SimpleCov.formatters = [
  SimpleCov::Formatter::HTMLFormatter,
  SimpleCov::Formatter::JSONFormatter,
  SimpleCov::Formatter::RcovFormatter
]

SimpleCov.start do
  add_filter '/spec/'
  add_group 'lib', 'lib'
end

require 'tensorflow'

# Requires supporting ruby files with custom matchers and macros, etc,
# in spec/support/ and its subdirectories.
Dir[File.dirname(__FILE__) + "/support/*.rb"].each {|f| require f }
