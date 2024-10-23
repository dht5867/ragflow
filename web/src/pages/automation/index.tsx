import { useNavigateWithFromState } from '@/hooks/route-hook';
import React, { useEffect } from 'react';
import { useLocation } from 'umi';
import styles from './index.less'; // Add styles if needed

const AutomationOps: React.FC = () => {
  const navigate = useNavigateWithFromState();
  const { pathname } = useLocation();

  useEffect(() => {
    if (pathname !== '/automation') {
      navigate('/automation');
    }
  }, [pathname, navigate]);

  return (
    <div className={styles.iframeContainer}>
      <iframe
        src="http://10.171.248.164:8502/" // The URL to load
        title="自动化运维"
        width="100%"
        height="92%"
        style={{ border: 'none', overflow: 'hidden' }} // Style to remove iframe border
      />
    </div>
  );
};

export default AutomationOps;
