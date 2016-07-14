Gem::Specification.new do |s|
  s.name        = 'tensorflow'
  s.version     = '0.0.1'
  s.date        = '2016-06-21'
  s.email       = 'arafat.da.khan@gmail.com'
  s.summary     = "A Machine learning gem for Ruby."
  s.description = "TensorFlow is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) that flow between them. "
  s.authors     = ["Arafat Dad Khan"]
  s.files       = `git ls-files -z`.split("\x0").reject { |f| f.match(%r{^(test|spec|features)/}) }
  s.license     = "Apache License, Version 2.0"
  s.homepage    = "TODO: qwewq lerwe"
  s.add_development_dependency "bundler", '~> 1.8', '>= 1.8.4'
  s.add_development_dependency "rake", "~> 10.0"
  s.add_development_dependency 'rspec', '~> 3.0'
  s.add_development_dependency 'google-protoc'
  s.extensions = ['ext/sciruby/tensorflow_c/extconf.rb']
end
