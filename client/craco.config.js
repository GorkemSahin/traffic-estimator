const CracoLessPlugin = require('craco-less');

module.exports = {
  plugins: [
    {
      plugin: CracoLessPlugin,
      options: {
        lessLoaderOptions: {
          lessOptions: {
            modifyVars: {
              '@primary-color': '#1db954',
              '@highlight-color': '#1db954',
              '@text-color': '#1db954',
              '@label-color': '#dfe4ea',
              '@input-color': 'black'
            },
            javascriptEnabled: true,
          },
        },
      },
    },
  ],
};